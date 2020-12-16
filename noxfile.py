import nox

nox.options.sessions = "lint", "tests"
locations = "clearbox_wrapper", "tests", "noxfile.py"


@nox.session(python=["3.8.5"])
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python=["3.8.5"])
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black", "flake8-bugbear", "flake8-import-order")
    session.run("flake8", *args)


@nox.session(python=["3.6.2", "3.7.0", "3.8.0", "3.8.5"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
