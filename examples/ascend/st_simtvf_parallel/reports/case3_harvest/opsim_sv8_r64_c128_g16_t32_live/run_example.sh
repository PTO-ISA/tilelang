#!/usr/bin/env bash
set +e
"${PTOAS_SIM_DSL_PYTHON_BIN}" "${PTOAS_SIM_DSL_EXAMPLE_PATH}" "$@"
status=$?
printf '%s\n' "${status}" > "${PTOAS_SIM_DSL_EXIT_CODE_FILE}"
exit "${status}"
