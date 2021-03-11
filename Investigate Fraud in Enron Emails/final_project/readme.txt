Project: Investigate Fraud in Enron Emails

Richard Smith
March 2021

readme.txt

Python version used:
  Python 3.8.5

---

Project files:
  /final_project/poi_id.py
    Code comprising data exploration, feature creation, feature selection,
      and machine learning algorithm used to fulfill project requirements.
    requires:
      /tools/feature_format.py
      /final_project/tester.py
      /final_project/final_project_dataset.pkl
  /final_project/Free-Response-Answers.html
    Documents project work via responses to assigned questions.
  /final_project/readme.txt
    This document, describes components of project submission.
  /final_project/references.txt
    Describes 3rd-party code references and material reviewed to complete
      project, excepting some references provided in this document for the git
      repo used as a base for project completion--a fork of Udacity's project
      repo created to migrate code to Python 3.8.5, which I forked for this
      submission.

---

Environment setup:

If using an Anaconda environment, please make use of python-3.8.yaml in the
  'environments' directory to set up an environment compatible with all code
  used. If 'pip' is preferred, 'requirements.txt' in the default directory of
  the repo may be used for the same.
Otherwise, please review those files in the case of any compatibility issues
  related to scikit-learn and all other related packages. 

---

Explanation for use of Python 3.8.5:

In attempting to complete the coursework leading up to this project, I first
  tried using Python 2.7 and the package specification listed in the
  'requirements.txt' present in the github repo provided by Udacity.
The following section describes the problems I ran into then, and my course
  of action afterward. I wrote all this as documentation in order to share
  it with other WGU students, in the hopes it would save them any headaches.

---

First, project instructions direct students to use Python 2.7 and to make use
  of a git repo for miniprojects/course project code:
  https://github.com/udacity/ud120-projects.git

Unfortunately, there are some issues with that repo's package requirements.
  Contents of requirements.txt:
    "nltk==3.2.1
     numpy==1.11.2
     scikit-learn==0.18
     scipy==0.18.1:"

Given these required packages, attempting to set up an Anaconda environment
  will result in conflicts: two cannot be installed for a Python 2.7
  environment, and the other two are unavailable via Anaconda repositories for
  a windows 64-bit OS. Attempting to install these via 'pip' results in
  somewhat complicated dependency issues (long, long error messages) which
  seem to also be related to Python 2.7 deprecation. Otherwise, my assumption
  is that these conflicts are based in my using Windows 10.

I'm sure that setting up a Python 2.7 Anaconda environment with these packages
  used to be trivial. but the best I could manage due to dependencies for other
  packages and conflicts was:
    nltk          3.2.1
    numpy         1.16.6
    scikit-learn  0.20.3
    scipy         1.2.1

For reference, here's the total output of 'conda list' for that environment,
  which includes all packages installed as dependencies for those listed just
  now:
  # packages in environment at C:\tools\Anaconda3\envs\py27:
  #
  # Name                    Version                   Build  Channel
  backports                 1.0                pyhd3eb1b0_2
  backports.functools_lru_cache 1.6.1              pyhd3eb1b0_0
  backports_abc             0.5                        py_1
  blas                      1.0                         mkl
  ca-certificates           2021.1.19            haa95532_0
  certifi                   2020.6.20          pyhd3eb1b0_3
  cycler                    0.10.0           py27h59acbbf_0
  freetype                  2.9.1                h4d385ea_1
  functools32               3.2.3.2                  py27_1
  futures                   3.3.0                    py27_0
  icc_rt                    2019.0.0             h0cc432a_1
  icu                       58.2                 h2aa20d9_1
  intel-openmp              2020.2                      254
  jpeg                      9b                   ha175dff_2
  kiwisolver                1.1.0            py27hc56fc5f_0
  libpng                    1.6.37               h7a46e7a_0
  matplotlib                2.2.3            py27h263d877_0
  mkl                       2020.2                      256
  mkl-service               2.3.0            py27h0b88c2a_0
  mkl_fft                   1.0.15           py27h44c1dab_0
  nltk                      3.2.1                    pypi_0    pypi
  numpy                     1.16.6           py27hcd21bde_0
  numpy-base                1.16.6           py27hb1d0314_0
  openssl                   1.0.2u               h0c8e037_0
  pip                       19.3.1                   py27_0
  pyparsing                 2.4.7              pyhd3eb1b0_0
  pyqt                      5.6.0            py27h6e61f57_6
  python                    2.7.18               hfb89ab9_0
  python-dateutil           2.8.1              pyhd3eb1b0_0
  pytz                      2021.1             pyhd3eb1b0_0
  qt                        5.6.2            vc9hc26998b_12
  scikit-learn              0.20.3           py27hf381715_0
  scipy                     1.2.1            py27h4c3ab11_0
  setuptools                44.0.0                   py27_0
  singledispatch            3.4.0.3                 py_1001
  sip                       4.18.1           py27hc56fc5f_2
  six                       1.15.0             pyhd3eb1b0_0
  sqlite                    3.30.1               h0c8e037_0
  tornado                   5.1.1            py27h0c8e037_0
  vc                        9                    h7299396_1
  vs2008_runtime            9.00.30729.1         hfaea7d5_1
  wheel                     0.36.2             pyhd3eb1b0_0
  wincertstore              0.2              py27hf04cefb_0
  zlib                      1.2.11               h3cc03e0_3

This setup permitted completion of some small portion of the course's material,
  but by the time I got along to a portion which made generation of plot
  windows via matplotlib necessary, popup-window generation via 'qt' failed,
  and my patience ran out.

Maybe those matplotlib.pyplot.show() call errors were related to the versions
  of pyqt or qt and how they interacted with the IDEs I tried (Sublime, VScode,
  and I'll beg you to believe that I set up their environment usage correctly
  and have used either for many, many projects), maybe the version for sip,
  maybe any number of combinations of dependencies interacting via functools32
  backport from Python 3.2 to 2.7, since functools32 was deemed	necessary by
  the environment dependency solver.

I'm sure much of the final project for the course can be completed with a
  Python 2.7 base, and that any number of workarounds may exist for the kinds
  of errors I encountered. Troubleshooting package dependency issues does not
  appeal to me, though, and that kind of effort is necessarily outside the
  scope of the course's material.

After exploring Udacity's "Knowledge" Q&As, I came across one contracted
  mentor/project reviewer repeatedly answering questions about Python 2.7
  issues with his own issue and personal solution: since macOS Big Sur does not
  support Python 2.7 at all, anymore, he's made his own *unofficial* update to
  the project's codebase in order to grade students' work:
    https://medium.com/udacity-course-companion/detecting-enron-fraud-in-macos-big-sur-edd2309f7389
    https://github.com/oforero/ud120-projects/tree/python-3.8

This seemed like a good starting point for attempting to complete the
  coursework and project with a newer version of Python, as contributors to
  that repo have been updating the code to reflect changes in the libraries
  involved, mitigating a wide variety of issues cited in Udacity "Knowledge"
  questions. Let it be known that no changes involving completion of any
  student-assigned code work have been made, to my knowledge or otherwise.

Of course, there were still some issues with the packages used in that project,
  as well. He suggests creating a Python 3.8 Anaconda environment by way of
  'python-3.8.yaml', a specification file which can be used in this manner:
    'conda env create environments/python-3.8.yaml'

Unfortunately, that syntax fails for my OS, and I had to use:
  'conda env create f=python-3.8.yaml'
(directly from 'environments' directory of the cloned repo, in this case)

At that point, environment creation failed due to these packages not being
  available for windows 64-bit systems via Anaconda repositories:
    libcxx=10.0.0
    libedit=3.1.20191231
    libffi=3.3
    libgfortran=3.0.1
    llvm-openmp=10.0.0
    ncurses=6.2
    readline=8.0

Makes sense, given what some of those are. Removing those lines from
  python-3.8.yaml, a wide variety of package conflicts resulted, but those were
  mitigated by further removal of these lines:
    mkl=2019.4
    pip=20.3.3
    wheel=0.36.2

After those changes, the environment was created successfully. Testing a
	miniproject script that relies on generating matplotlib.pyplot.show() windows
	resulted in success, visible plots.

For reference, here's the total output of 'conda list' for that environment:
  # packages in environment at C:\tools\Anaconda3\envs\py38:
  #
  # Name                    Version                   Build  Channel
  backcall                  0.2.0              pyhd3eb1b0_0
  blas                      1.0                         mkl
  ca-certificates           2021.1.19            haa95532_0
  certifi                   2020.12.5        py38haa95532_0
  click                     7.1.2              pyhd3eb1b0_0
  colorama                  0.4.4              pyhd3eb1b0_0
  cycler                    0.10.0                   py38_0
  decorator                 4.4.2              pyhd3eb1b0_0
  freetype                  2.10.4               hd328e21_0
  icc_rt                    2019.0.0             h0cc432a_1
  icu                       58.2                 ha925a31_3
  intel-openmp              2019.4                      245
  ipykernel                 5.3.4            py38h5ca1d4c_0
  ipython                   7.18.1                   py38_0    esri
  ipython_genutils          0.2.0              pyhd3eb1b0_1
  jedi                      0.16.0                   py38_0    esri
  joblib                    1.0.0              pyhd3eb1b0_0
  jpeg                      9b                   hb83a4c4_2
  jupyter_client            6.1.7                      py_0    esri
  jupyter_core              4.6.3                    py38_2    esri
  kiwisolver                1.3.0            py38hd77b12b_0
  lcms2                     2.11                 hc51a39a_0
  libpng                    1.6.37               h2a8f88b_0
  libsodium                 1.0.18                        1    esri
  libtiff                   4.1.0                h56a325e_1
  lz4-c                     1.9.2                hf4a77e7_3
  matplotlib                3.3.2                haa95532_0
  matplotlib-base           3.3.2            py38hba9282a_0
  mkl                       2020.2                      256
  mkl-service               2.3.0            py38h196d8e1_0
  mkl_fft                   1.2.0            py38h45dec08_0
  mkl_random                1.1.1            py38h47e9c7a_0
  nltk                      3.5                        py_0
  numpy                     1.19.2           py38hadc3359_0
  numpy-base                1.19.2           py38ha3acd2a_0
  olefile                   0.46                       py_0
  openssl                   1.1.1i               h2bbff1b_0    esri
  pandas                    1.1.5            py38hf11a4ad_0
  parso                     0.8.1              pyhd3eb1b0_0
  pickleshare               0.7.5           pyhd3eb1b0_1003
  pillow                    8.0.1            py38h4fa10fc_0
  pip                       20.3.3           py38haa95532_0
  prompt_toolkit            3.0.5                      py_0    esri
  pygments                  2.7.0                      py_0    esri
  pyparsing                 2.4.7              pyhd3eb1b0_0
  pyqt                      5.9.2            py38ha925a31_4
  python                    3.8.5                h5fd99cc_1
  python-dateutil           2.8.1              pyhd3eb1b0_0
  pytz                      2020.4             pyhd3eb1b0_0
  pyzmq                     19.0.2                   py38_1    esri
  qt                        5.9.7            vc14h73c81de_0
  regex                     2020.11.13       py38h2bbff1b_0
  scikit-learn              0.23.2           py38h47e9c7a_0
  scipy                     1.5.2            py38h14eb087_0
  seaborn                   0.11.1             pyhd3eb1b0_0
  setuptools                51.0.0           py38haa95532_2
  sip                       4.19.13          py38ha925a31_0
  six                       1.15.0           py38haa95532_0
  sqlite                    3.33.0               h2a8f88b_0
  threadpoolctl             2.1.0              pyh5ca1d4c_0
  tk                        8.6.10               he774522_0
  tornado                   6.1              py38h2bbff1b_0
  tqdm                      4.56.0             pyhd3eb1b0_0
  traitlets                 5.0.5              pyhd3eb1b0_0
  vc                        14.2                 h21ff451_1
  vs2015_runtime            14.27.29016          h5e58377_2
  wcwidth                   0.2.5                      py_0
  wheel                     0.36.2             pyhd3eb1b0_0
  wincertstore              0.2                      py38_0
  xz                        5.2.5                h62dcd97_0
  zeromq                    4.3.2                         2    esri
  zlib                      1.2.11               h62dcd97_4
  zstd                      1.4.5                h04227a9_0

Let it be known that nltk is not specified in python-3.8.yaml, and that it
  and these dependencies were installed separately:
    nltk    3.5
    click   7.1.2
    regex   2020.11.13
    tqdm    4.56.0

nltk is used in coursework, but is not mandatory for the final project.

---