# BB-AWG-SCRIPTS
## Purpose of this repository

This repository is for storing various pipelines, ad-hoc scripts, specific configuration
files, and notes created to accomplish BB-AWG-related tasks.  "Tasks"
should be taken broadly.

This is meant to achieve a few goals:

- Facilitate sharing.  Like a shared filesystem.
- Establish a record.  For traceable, reproducible results.
- Release pressure on the more "traditional" repositories.  Because
  ad-hoc work is stored here, the main codebases can be kept clean of
  such things.


## How to use this repository

This is a shared workspace, so things will get really ugly unless we
follow some rules:

- Respect the organizational structure described below.
- Do not commit large (> 1MB) data files.  Everyone will like it
  better if you include a link or download instructions in a local
  script or README.
- Do not commit the plots or other output from your scripts.  Those
  should be communicated through some other means (wiki, webspace, presentation, etc,.)

Note you can use ``.gitignore`` files to help with this -- for
example, consider adding ``*.png`` to ``.gitignore`` in
``/users/you/.gitignore``.

##  Organization structure
```
  /users
         /lowercaseusername
         /lowercaseusername2
         ...
  /pipelines
         /iso_sim_ver002
         /iso
         /misc
         ...
  /misc
```
