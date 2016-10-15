# scipy-articles
This repository is meant for storing and collaborating on publications about
SciPy written by the SciPy development team.

**Repo organization**: one independent directory per article.  Within that
directory should be placed:

- all article content (text, code, supporting data)
- tests for code if needed
- build scripts, style files, etc. to reproduce the complete article
- a README file to explain how to build the article and details on where it has
  been or will be submitted, who are the lead authors, etc.

**License**: all content in this repo is BSD-licensed.  Using the same license
as for SciPy itself helps in reusing code, documentation and text.
Some content may have to be dual-licensed to conform to journal requirements
(for example: The Journal of Open Source Software) mandates CC BY 4.0 for text
and MIT for code).  This will be made clear in the README of the article
itself.

**Governance**: decisions about content in this repository, and about which
articles to write in the first place, are made by the SciPy development team in
the same way as decisions about SciPy itself are made.

## Types of publications

At the moment we foresee writing two kinds of articles:

1. Traditional full-length articles (peer reviewed).
2. Abstract articles (peer reviewed).

An abstract article is a very brief article, aimed at providing a citable
peer-reviewed paper with minimal effort that allows contributors to get
academic credit for their work.  See http://joss.theoj.org/about for more
details on this.

The purposes of these two types of articles are different.  An abstract article
is only meant to give credit to contributors to (a) particular release(s).
The frequency of writing such an article is a balance between providing new
contributors with credit and diluting citations.  A reasonable frequency is
probably once every 1.5 or 2 years.

A full-length article contains content that will help readers understand the
scope of SciPy, the development process, major new features, plans for future
development, etc.  It also provides a more significant level of credit, hence
the higher bar for authorship (see below).  We plan only a single such article
at the moment, with the publication date coinciding with the 1.0 release.

## Authorship policy

We aim to err on the side of being too inclusive rather than exclusive.
Any concerns regarding the author list for a particular article can be brought
up on the scipy-dev mailing list or directly with one of the authors or SciPy
core developers.

We believe it's not productive to define a criterion for inclusion as an author
on a paper in terms of number of commits or lines of code written.  Instead, we
will use these guidelines:

- For a full-length article, an author should have contributed at least either
  one major feature in SciPy or done a significant amount of maintenance work.
- For an abstract article, an author should have made a nontrivial
  contribution.  This means: fixing a typo is not enough, but a single bug fix
  can be.

The author list for an abstract article will be ordered alphabetically or
reverse-alphabetically (alternating, starting with alphabetically).

The author list for a full article will be ordered by commit count over the
period the paper covers.

