import nox


@nox.session(python=["3.7.0", "3.8.0", "3.8.5"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
