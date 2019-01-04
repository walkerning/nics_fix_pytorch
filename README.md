## Fixed Point Training Simulation Framework on PyTorch

### Core Functionality
- Parameter/Buffer fix: by using `nics_fix_pt.nn_fix.<original name>_fix` modules;
  parameters are those to be trained, buffers are those to be persistenced but not considered parameter
- Activation fix: by using `ActivationFix` module
- Data fix VS. Gradient fix: by supply `nf_fix_params`/`nf_fix_params_grad` kwargs args
      in `nics_fix_pt.nn_fix.<original name>_fix` or `ActivationFix` module construction

### Utilities
- FixTopModule: dump/load fix configuration to/from file; print fix configs. 

  FixTopModule is just a wrapper that gather config print/load/dump/setting utilities, these utilities will work with nested normal nn.Module as intermediate module containers, e.g. `nn.Sequential` of fixed modules will also work, you do not need to have a subclass multi-inherited from nn.Sequential and nfp.FixTopModule!

> NOTE: parameters are saved as float, and in most use cases(e.g. fixed-point hardware simultation), you always need to dump and then load/modify the fixed configurations of the variables using `model.get_fix_configs` and `model.load_fix_configs`. Check `examples/mnist/train_mnist.py` for an example.

### Examples
See `examples/mnist/train_mnist.py` for a MNIST fix-point training example.

### Test cases

![coverage percentage](./coverage.svg)

Run `python setup.py test` to run the pytest test cases.
