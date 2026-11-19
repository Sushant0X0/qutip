"""
Diagnostics for qutip issue #3104: does ``dnorm`` fail with cvxpy/CVXOPT on
this machine, and does it depend on the BLAS/LAPACK cvxopt links against or on
solver settings?

It reproduces ``Test_dnorm.test_cp_case`` (the test that fails on macOS ARM)
plus one boundary case, over many seeds, for several solver configurations.
qutip's ``dnorm`` cannot forward solver options, so the SDP solve is done here
in the same way ``dnorm(..., force_solve=True)`` does it.

Usage:
    python diagnose_dnorm_3104.py [--seeds 50] [--verbose-first-failure]
"""
import argparse
import importlib
import platform
import subprocess
import sys
import traceback

import numpy as np
import scipy
import scipy.sparse as sp

import qutip
from qutip import rand_super_bcsz, to_choi, qeye, sigmax, dnorm
from qutip.core import data as _data
from qutip.core.semidefinite import dnorm_problem, dnorm_sparse_problem

try:
    import cvxpy
except ImportError:
    sys.exit("cvxpy is not installed; nothing to diagnose.")


def env_report():
    print("=" * 72)
    print("platform :", platform.platform(), platform.machine())
    print("python   :", sys.version.replace("\n", " "))
    for name in ["numpy", "scipy", "qutip", "cvxpy", "cvxopt", "scs",
                 "clarabel"]:
        try:
            mod = importlib.import_module(name)
            print(f"{name:9s}:", getattr(mod, "__version__", "?"),
                  "-", mod.__file__)
        except ImportError:
            print(f"{name:9s}: not installed")
    print("cvxpy solvers:", cvxpy.installed_solvers())

    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg["Build Dependencies"]["blas"]
        print("numpy BLAS   :", blas.get("name"), blas.get("version"))
    except Exception:
        pass

    # Which BLAS/LAPACK does cvxopt actually link against?
    try:
        import cvxopt.lapack
        so = cvxopt.lapack.__file__
        tool = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
        out = subprocess.run(tool + [so], capture_output=True, text=True)
        print("cvxopt.lapack linkage:")
        print(out.stdout or out.stderr)
    except Exception as exc:
        print("could not inspect cvxopt linkage:", exc)
    print("=" * 72)


def build_problem(J, sparse):
    """Same construction as qutip.core.metrics.dnorm's SDP branch."""
    dim = int(np.prod(J.dims[0][0]))
    J_dat = _data.to("csr", J.data).as_scipy()
    if not sparse:
        problem, Jr, Ji = dnorm_problem(dim)
        Jr.value = sp.csr_array(
            (J_dat.data.real, J_dat.indices, J_dat.indptr), shape=J_dat.shape
        ).toarray()
        Ji.value = sp.csr_array(
            (J_dat.data.imag, J_dat.indices, J_dat.indptr), shape=J_dat.shape
        ).toarray()
    else:
        problem = dnorm_sparse_problem(dim, J_dat)
    return problem


def sdp_dnorm(J, sparse, solver, opts, verbose=False):
    problem = build_problem(J, sparse)
    problem.solve(solver=solver, verbose=verbose, **opts)
    return problem.value, problem.status


def configurations():
    installed = set(cvxpy.installed_solvers())
    confs = []
    if "CVXOPT" in installed:
        confs += [
            ("CVXOPT default", "CVXOPT", {}),
            ("CVXOPT kktsolver=robust", "CVXOPT", {"kktsolver": "robust"}),
            ("CVXOPT refinement=3", "CVXOPT", {"refinement": 3}),
        ]
    for name in ["CLARABEL", "SCS"]:
        if name in installed:
            confs.append((name, name, {}))
    return confs


def cases(n_seeds):
    """(label, Choi matrix, reference value or None)."""
    out = []
    # Test_dnorm.test_cp_case: CP (not TP) maps have a closed form,
    # which dnorm uses when force_solve=False.
    for dim in (2, 3):
        for seed in range(n_seeds):
            A = rand_super_bcsz(dim, enforce_tp=False, seed=seed)
            out.append((f"cp dim={dim} seed={seed}", to_choi(A), dnorm(A)))
    # Boundary case from test_qubit_simple_known_cases: ||id - X|| = 2.
    out.append(("id - X", to_choi(qeye(2)) - to_choi(sigmax()), 2.0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="abs tolerance used by test_cp_case")
    parser.add_argument("--verbose-first-failure", action="store_true")
    args = parser.parse_args()

    env_report()
    all_cases = cases(args.seeds)
    print(f"{len(all_cases)} cases x sparse/dense\n")

    shown_verbose = False
    for label, solver, opts in configurations():
        n_err = n_bad = n_ok = 0
        worst = 0.0
        first_failure = None
        for case, J, ref in all_cases:
            for sparse in (True, False):
                try:
                    val, status = sdp_dnorm(J, sparse, solver, opts)
                except cvxpy.error.SolverError:
                    n_err += 1
                    first_failure = first_failure or (case, J, sparse)
                    continue
                except Exception:
                    n_err += 1
                    print(f"  [{label}] {case}: unexpected exception")
                    traceback.print_exc()
                    continue
                err = abs(val - ref) if val is not None else np.inf
                worst = max(worst, err)
                if err > args.tol or status != "optimal":
                    n_bad += 1
                    first_failure = first_failure or (case, J, sparse)
                else:
                    n_ok += 1
        print(f"{label:26s} ok={n_ok:4d}  SolverError={n_err:4d}  "
              f"inaccurate/not-optimal={n_bad:4d}  worst abs err={worst:.2e}")

        if args.verbose_first_failure and first_failure and not shown_verbose:
            case, J, sparse = first_failure
            print(f"\n--- verbose re-run of first failure: {case}, "
                  f"sparse={sparse}, {label} ---")
            try:
                sdp_dnorm(J, sparse, solver, opts, verbose=True)
            except cvxpy.error.SolverError as exc:
                print("SolverError:", exc)
            print("--- end verbose ---\n")
            shown_verbose = True


if __name__ == "__main__":
    main()
