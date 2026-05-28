# Contributing to Pandorica

Thanks for considering a contribution. Before opening a pull request, please
read this short document — especially the **Developer Certificate of Origin
(DCO)** section, which is required for every commit.

## License terms for contributions

`pandorica` is released under the
[PolyForm Noncommercial License 1.0.0](LICENSE). The maintainer also sells
separate commercial licenses (see [COMMERCIAL.md](COMMERCIAL.md)) to fund the
project.

For that dual-track model to work cleanly, the maintainer must be able to
license each contribution under both tracks. By signing off on your commits
(below), you agree that:

1. You **authored** the contribution, or you have the right to submit it
   under the project's license.
2. You grant the maintainer the right to redistribute your contribution
   under the project's current license (PolyForm Noncommercial 1.0.0) **and**
   under separate commercial licenses negotiated by the maintainer with
   third parties.
3. You understand the contribution and your sign-off are public, will be
   redistributed with the project indefinitely, and your name and email in
   the sign-off line will be retained in the public git history.

There is **no separate Contributor License Agreement (CLA)** to sign. The
DCO sign-off below is the agreement.

## Developer Certificate of Origin (DCO)

The DCO is a short, well-known statement from the Linux kernel and many
other major open-source projects. Read the full text at
<https://developercertificate.org/>. By adding `Signed-off-by:` to a commit
you certify the DCO for that commit.

### How to sign off

Append a `Signed-off-by:` line to every commit message, using the same name
and email as your git configuration:

```text
Implement feature X

Signed-off-by: Jane Doe <jane@example.com>
```

The easiest way is to use git's built-in `-s` flag:

```bash
git commit -s -m "Implement feature X"
```

You can also amend an existing commit to add the sign-off:

```bash
git commit --amend -s --no-edit
```

If a pull request contains commits without a sign-off, the maintainer will
ask you to amend them (or do a rebase with `git rebase --signoff`) before
merging.

## How to contribute

1. **Open an issue first** for non-trivial work. Bug reports are welcome
   without prior discussion; new features and architectural changes benefit
   from a short design conversation before code is written.
2. **Fork** the repository and create a topic branch off `main`.
3. **Keep changes focused.** One logical change per pull request makes
   review faster.
4. **Add or update tests.** The test suite must run without the large
   datasets (`@pytest.mark.data` covers data-dependent tests). Run
   `pytest` locally before pushing.
5. **Follow the existing style.** Run `black` and `flake8` (configured in
   `pyproject.toml` / `setup.cfg`) before committing.
6. **Sign off every commit** as described above.
7. **Open a pull request** against `main` with a clear description of what
   changed and why.

## What kinds of contributions are wanted

- Bug fixes with a regression test.
- Performance improvements with benchmarks.
- Documentation improvements, examples, and clearer error messages.
- Tests that exercise real-data edge cases (within `@pytest.mark.data`).
- New algorithmic stages — but please discuss in an issue first, since the
  project has a deliberate scope and a `Future work` list in
  `pandorica/stitch/README.md`.

## What's out of scope

- Vendoring third-party source into the tree (we prefer dependencies).
- Changes that relicense the project, remove the noncommercial restriction,
  or strip the `Required Notice:` lines from `LICENSE`.

## Contact

For licensing questions, see [COMMERCIAL.md](COMMERCIAL.md). For everything
else, open an issue or email <robert.kiewisz@gmail.com>.
