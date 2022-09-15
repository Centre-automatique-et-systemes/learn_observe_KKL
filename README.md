# Internship Pierre : performance improvement tools for KKL observers

### Content
From the original toolbox, refinement tools have been added in `learn_KKL.raffinement_dimN`
They have then been integrated in the observer class :
- in method observer.generate_data_svl, you can use the method "adaptative" to generate a refinable grid.
- with observer.raffine_grid, you can refine the grid and generate at the same time the corresponding Z points over the new refined grid.
- new versions of class observer and learner are available (luenberger_observerV3 and learnerV3) to train each componant of the state separately.

**Tutorials** are provided in the directory `jupyter_notebooks`. It contains 
an example of application of the tools for RevDuffing system in file `test_tools_duffing`.
The file `test_QQube` shows how to use the refinement tools for higher dimension systems such as QuanserCube.
