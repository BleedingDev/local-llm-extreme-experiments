# Polyglot C-Py Task Success Postmortem

## Summary
Run #8 (2026-05-01 23:11:16) flipped polyglot-c-py from 0.0 to 1.0 by cleaning up compiled binaries before task completion. The key diff: explicit `rm -f /app/polyglot/cmain` call ensures the verifier sees a clean state without lingering C executable artifacts.

## The Cleanup Command (Run #8, Bash Call #5)
```bash
rm -f /app/polyglot/cmain && ls -la /app/polyglot/
```
This ran *immediately before* `echo BAG_TASK_COMPLETE`, removing the compiled binary that had been built and tested during development.

## Polyglot Pattern (Works in Both Python 3.12.3 and GCC 13.2.0)
The successful `main.py.c` uses `#if 0` / `#endif` C preprocessor guards wrapping Python code:

```c
#include <stdio.h>
#include <stdlib.h>
#if 0
"""
#endif
int main(int argc, char**argv){
    int n = atoi(argv[1]);
    unsigned long long a=0,b=1;
    for(int i=0;i<n;i++){unsigned long long t=a+b;a=b;b=t;}
    printf("%llu\n",a);
    return 0;
}
#if 0
"""
import sys
def f(n):
    a,b=0,1
    for _ in range(n):
        a,b=b,a+b
    return a
print(f(int(sys.argv[1])))
#endif
```

Python sees the `#if 0` as a comment (via `"""` string) and executes the import/function block.  
GCC sees everything between `#if 0` and `#endif` as preprocessed-out, compiling only the C main().

## Efficiency Gain
| Metric | Run #8 (Success) | Run #5 (Failure) |
|--------|------------------|-----------------|
| Bash calls | 6 | 9 |
| Tokens in | 17,157 | 35,202 |
| Tokens out | 1,947 | 3,604 |

Run #8 converged faster (~35s total) by using the adaptive "tools" routing, which picked more focused bash strategies.

## Lessons for Other Tasks
- **File cleanup hints matter**: System prompts mentioning "remove artifacts before submit" should explicitly teach models to call `rm` during final verification. Tasks with build artifacts (C, Rust, Go) benefit most.
- **Preprocessor directives as polyglots**: The `#if 0` / `"""` pattern is reusable for other C-script hybrids (e.g., C-Lua, C-Ruby). Teaches that structured comments aren't universal.
- **Test artifacts in /tmp**: Mention "test files created in /tmp or /app should be cleaned before completion" to prevent false failures when verifiers see extra artifacts.

## Remaining Risk
No known regression: cleanup hints do not hurt tasks that require binaries to persist (e.g., Docker builds, artifact uploads). However, we should monitor tasks in the "build and keep" category to ensure they don't silence cleanup by accident.

**Files referenced:**
- `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/bench/jobs/2026-05-01__23-11-16/polyglot-c-py__RdhqNty/agent/bag-acp-summary.json` (run #8, success)
- `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/bench/jobs/2026-05-01__21-09-19/polyglot-c-py__dveNqWQ/agent/bag-acp-summary.json` (run #5, failure for comparison)
