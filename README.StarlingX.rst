StarlingX Nova stx/stein Branch
===============================

The stx/stein series of branches is created from upstream Nova stable/stein
branches to provide a place to backport additional work from Train master
development for StarlingX Nova containers.

The stx/stein branch will be periodically rebased on upstream stable/stein
as upstream changes so all of the additional commits that StarlingX backports
will be on top of a git histroy that matches upstream.

Rather than force push over a single stx/stein branch, we will create a new
branch with a numerically increasing suffix for each rebase.  This also allows
StarlingX to be able to continue to drop back to previous stx/stein branches
as required during its release cycle.

::

    Date        Branch          Upstream SHA/Tag
    --------------------------------------------
    2019-04-16  stx/stein.1     35398521
