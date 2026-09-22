"""Philox4x32-10 RNG helper for TileLang PTO SIMT kernels.

Mirrors the Ascend SIMT reference implementation
(src/tl_templates/ascend/philox_rng.h and random_kernel_base.h) so PTO and
AscendC runs share the same Philox counter/consumption rules and the same
Box-Muller normal transform.

The state layout matches tl::AscendPhiloxState (key/counter/buf/idx plus the
Box-Muller cache), but every field is a *runtime* per-lane value living in a
SIMT-local buffer, exactly like the C struct lives in per-lane storage on the
Ascend target. Draws therefore carry no trace-time Python state and are valid
in any control flow: straight-line code, device loops (``range`` /
``pto.for_(...)``, hinted or not), and runtime branches. The "generate a fresh
block every four draws" decision is a runtime ``pto.if_`` on the device, not a
Python ``if`` at trace time.
"""

from ptodsl import pto, scalar

# Philox4x32-10 constants, identical to random_kernel_base.h.
_PHILOX_M4X32_A = 0xD2511F53
_PHILOX_M4X32_B = 0xCD9E8D57
_PHILOX_W32_A = 0x9E3779B9
_PHILOX_W32_B = 0xBB67AE85

# Uniform mapping constants: u = rand() * 2^-32 + 2^-33.
_RAND_2POW32_INV = 2.3283064e-10
_RAND_2POW32_INV_HALF = 1.1641532182693481e-10

_TWO_PI = 6.2831854820251465  # 2*pi in float32
_NORMAL_EPS = 1.0e-7

# Runtime state layout inside the per-lane ui32 buffer (indices are
# compile-time constants, so accesses stay register-friendly).
_KEY0 = 0
_KEY1 = 1
_CTR0 = 2
_CTR1 = 3
_CTR2 = 4
_CTR3 = 5
_BUF0 = 6
_BUF1 = 7
_BUF2 = 8
_BUF3 = 9
_IDX = 10  # 0..4; 4 == exhausted, regenerate on next draw
_HAS_NORMAL = 11  # 1 when the Box-Muller spare is cached
_STATE_SLOTS = 12


def _i32(value):
    # Internal Philox state uses plain signless i32 values: pto.mulhi's
    # verifier rejects ui32 operands (signedness is an op attribute), and the
    # arithmetic below only relies on wrap-around semantics plus sign-agnostic
    # equality compares.
    return pto.const(value & 0xFFFFFFFF, dtype=pto.i32)


def _split_i64_bits(value):
    """Return the low/high i32 words of a Python or runtime 64-bit value."""
    if isinstance(value, int):
        return _i32(value), _i32(value >> 32)

    value64 = scalar.cast(value, pto.i64)
    low = scalar.cast(value64, pto.i32)
    high = pto.mulhi(
        value64,
        pto.const(1 << 32, dtype=pto.i64),
        signedness="unsigned",
    )
    return low, scalar.cast(high, pto.i32)


class PhiloxRNG:
    """Per-lane Philox4x32-10 state for PTO SIMT kernels.

    ``seed``, ``seq``, and ``off`` may be Python integers or runtime integer
    scalars. ``off`` follows the nonnegative int64 contract of the Ascend
    reference and is converted to a Philox block offset with ``ceil(off / 4)``.

    All state is stored in per-lane SIMT-local memory at construction time, so
    the object must be created inside a ``pto.simt`` section (the TileLang PTO
    codegen emits it inside ``T.SimtVF``).
    """

    def __init__(self, seed, seq, off=0):
        if isinstance(seed, int) and not 0 <= seed < (1 << 64):
            raise ValueError("PhiloxRNG seed must fit in 64 bits")
        if isinstance(off, int) and not 0 <= off < (1 << 64):
            raise ValueError("PhiloxRNG off must fit in 64 bits")

        # Internal slots use signless i32 semantics (see _i32); rand()
        # bitcasts to ui32 only at the boundary.
        self._state = pto.alloc_buffer((_STATE_SLOTS,), pto.i32)
        self._normal_cache = pto.alloc_buffer((1,), pto.f32)

        key0, key1 = _split_i64_bits(seed)
        # Counter starts at zero; SkipLo adds ceil(off/4) to the low 64 bits
        # and SkipHi places seq in the high 64 bits. Adding to zero counters
        # cannot carry, so the initial values are direct assignments.
        block_offset = (off + 3) // 4
        ctr0, ctr1 = _split_i64_bits(block_offset)
        ctr2, ctr3 = _split_i64_bits(seq)

        scalar.store(key0, self._state, _KEY0)
        scalar.store(key1, self._state, _KEY1)
        scalar.store(ctr0, self._state, _CTR0)
        scalar.store(ctr1, self._state, _CTR1)
        scalar.store(ctr2, self._state, _CTR2)
        scalar.store(ctr3, self._state, _CTR3)
        # Buffer contents are irrelevant while idx == 4 (forces generation on
        # the first draw), but keep the state fully initialized anyway.
        for slot in (_BUF0, _BUF1, _BUF2, _BUF3):
            scalar.store(_i32(0), self._state, slot)
        scalar.store(_i32(4), self._state, _IDX)
        scalar.store(_i32(0), self._state, _HAS_NORMAL)

    @staticmethod
    def _mulhi(a, b):
        return pto.mulhi(a, b, signedness="unsigned")

    def _round(self, c0, c1, c2, c3, k0, k1):
        mul_a = _i32(_PHILOX_M4X32_A)
        mul_b = _i32(_PHILOX_M4X32_B)
        lo0 = mul_a * c0
        hi0 = self._mulhi(mul_a, c0)
        lo1 = mul_b * c2
        hi1 = self._mulhi(mul_b, c2)
        return hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0

    def _generate(self):
        # PhiloxRandomSimt: 10 rounds on temporaries; the key is bumped after
        # every round. State slots are updated only after the block finishes.
        c0 = scalar.load(self._state, _CTR0)
        c1 = scalar.load(self._state, _CTR1)
        c2 = scalar.load(self._state, _CTR2)
        c3 = scalar.load(self._state, _CTR3)
        k0 = scalar.load(self._state, _KEY0)
        k1 = scalar.load(self._state, _KEY1)
        for _ in range(10):
            c0, c1, c2, c3 = self._round(c0, c1, c2, c3, k0, k1)
            k0 = k0 + _i32(_PHILOX_W32_A)
            k1 = k1 + _i32(_PHILOX_W32_B)
        scalar.store(c0, self._state, _BUF0)
        scalar.store(c1, self._state, _BUF1)
        scalar.store(c2, self._state, _BUF2)
        scalar.store(c3, self._state, _BUF3)

    def _skip_one(self):
        # 128-bit increment of (ctr0..ctr3) with explicit carry propagation,
        # matching SkipOne in the Ascend reference. Written branch-free:
        # counter values are runtime scalars, so data-dependent Python branches
        # cannot be traced.
        one = _i32(1)
        zero = _i32(0)
        ctr0 = scalar.load(self._state, _CTR0) + one
        scalar.store(ctr0, self._state, _CTR0)
        k1 = scalar.select(ctr0 == 0, one, zero)
        ctr1 = scalar.load(self._state, _CTR1) + k1
        scalar.store(ctr1, self._state, _CTR1)
        k2 = scalar.select(ctr1 == 0, k1, zero)
        ctr2 = scalar.load(self._state, _CTR2) + k2
        scalar.store(ctr2, self._state, _CTR2)
        k3 = scalar.select(ctr2 == 0, k2, zero)
        ctr3 = scalar.load(self._state, _CTR3) + k3
        scalar.store(ctr3, self._state, _CTR3)

    def _rand_i32(self):
        # Regenerate the block when exhausted: a runtime branch on the device,
        # so the draw works inside device loops and runtime branches alike.
        idx = scalar.load(self._state, _IDX)
        with pto.if_(idx >= 4) as br, br.then_:
            self._generate()
            self._skip_one()
            scalar.store(_i32(0), self._state, _IDX)
        idx = scalar.load(self._state, _IDX)
        # Pick the buffer slot with a select tree: the slot index is a runtime
        # value, while buffer accesses here take compile-time-constant offsets.
        buf0 = scalar.load(self._state, _BUF0)
        buf1 = scalar.load(self._state, _BUF1)
        buf2 = scalar.load(self._state, _BUF2)
        buf3 = scalar.load(self._state, _BUF3)
        out = scalar.select(
            idx == 0,
            buf0,
            scalar.select(idx == 1, buf1, scalar.select(idx == 2, buf2, buf3)),
        )
        scalar.store(idx + 1, self._state, _IDX)
        return out

    def rand(self):
        """Draw one raw uint32 from the lane's stream, advancing the state."""
        # Reinterpret as ui32 at the boundary so the value matches uint32
        # buffers; the bit pattern is unchanged.
        return scalar.cast(self._rand_i32(), pto.ui32)

    def rand_uniform(self):
        """Draw a float32 uniformly distributed in [0, 1)."""
        value = pto.convert(
            self._rand_i32(),
            pto.f32,
            rounding="r",
            saturation="nosat",
            signedness="unsigned",
        )
        return value * _RAND_2POW32_INV + _RAND_2POW32_INV_HALF

    def rand_normal(self):
        """Draw a float32 from N(0, 1) via Box-Muller (two draws cached)."""
        if not hasattr(pto, "sin") or not hasattr(pto, "cos"):
            raise RuntimeError(
                "PhiloxRNG.rand_normal requires pto.sin/pto.cos (A5 SIMT "
                "sin/cos SoftLib, PTOAS PR #1193); the current ptodsl build "
                "does not provide them"
            )
        has_normal = scalar.load(self._state, _HAS_NORMAL)
        with pto.if_(has_normal != 0) as br:
            with br.then_:
                scalar.store(_i32(0), self._state, _HAS_NORMAL)
                br.assign(out=scalar.load(self._normal_cache, 0))
            with br.else_:
                u1 = pto.fmax(self.rand_uniform(), _NORMAL_EPS)
                u2 = self.rand_uniform()
                r = pto.sqrt(pto.log(u1) * -2.0)
                v = u2 * _TWO_PI
                z0 = r * pto.sin(v)
                z1 = r * pto.cos(v)
                scalar.store(z1, self._normal_cache, 0)
                scalar.store(_i32(1), self._state, _HAS_NORMAL)
                br.assign(out=z0)
        return br.out
