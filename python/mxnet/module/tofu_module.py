import logging

from .base_module import BaseModule

from .. import context as ctx
from .. import ndarray as nd
from .. import optimizer as opt

from ..base import mx_real_t
from ..model import _update_params
from ..model import load_checkpoint
from ..initializer import Uniform, InitDesc
from ..io import DataDesc

class TofuModule(BaseModule):
    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu()):
        super(TofuModule, self).__init__(logger=logger)
        if isinstance(context, ctx.Context):
            context = [context]
        self._context = context
        if len(self._context) == 1:
            self._default_context = self._context[0]
        else:
            self._default_context = ctx.cpu()

        self._symbol = symbol

        data_names = list(data_names)
        label_names = list(label_names) if label_names is not None else []

        arg_names = symbol.list_arguments()
        input_names = data_names + label_names
        self._param_names = [x for x in arg_names if x not in input_names]
        self._aux_names = symbol.list_auxiliary_states()
        self._data_names = data_names
        self._label_names = label_names
        self._output_names = symbol.list_outputs()

        print('Data & label names:', input_names)
        print('Parameter names:', self._param_names)
        print('Auxiliary names:', self._aux_names)

        self._arg_params = None
        self._aux_params = None
        self._params_dirty = False

        self._optimizer = None
        self._updater = None
        self._preload_opt_states = None
        self._grad_req = None

        self._binded = False
        self._exec = None
        self._data_desc = None
        self._label_desc = None

        self._params_initialized = False
        self._optimizer_initialized = False

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        sym, args, auxs = load_checkpoint(prefix, epoch)
        mod = TofuModule(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod._params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states'%(prefix, epoch)
        return mod

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        self._symbol.save('%s-symbol.json'%prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        logging.info('Saved checkpoint to \"%s\"', param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self.save_optimizer_states(state_name)
            logging.info('Saved optimizer state to \"%s\"', state_name)

    def _reset_bind(self):
        """Internal function to reset binded state."""
        self._binded = False
        self._exec = None
        self._data_desc = None
        self._label_desc = None

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    @property
    def data_shapes(self):
        """Get data shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self._binded
        return self._data_desc

    @property
    def label_shapes(self):
        """Get label shapes.
        Returns
        -------
        A list of `(name, shape)` pairs. The return value could be `None` if
        the module does not need labels, or if the module is not binded for
        training (in this case, label information is not available).
        """
        assert self._binded
        return self._label_desc

    @property
    def output_shapes(self):
        """Get output shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self._binded
        # TODO(minjie)
        raise NotImplementedError()

    def get_params(self):
        assert self._binded and self._params_initialized
        if self._params_dirty:
            self._sync_params_from_devices()
        return (self._arg_params, self._aux_params)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        if self._params_initialized and not force_init:
            return
        assert self._binded, 'call bind before initializing the parameters'

        if self._arg_params is None:
            self._arg_params = {name: nd.zeros_like(self._exec.arg_dict[name])
                                for name in self._param_names}

        if self._aux_params is None:
            self._aux_params = {name: nd.zeros_like(self._exec.aux_dict[name])
                                for name in self._aux_names}

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        cache_arr.copyto(arr)
                else:
                    if not allow_missing:
                        raise RuntimeError("%s is not presented" % name)
                    if initializer != None:
                        initializer(name, arr)
            else:
                initializer(name, arr)

        attrs = self._symbol.attr_dict()
        for name, arr in self._arg_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, arg_params)

        for name, arr in self._aux_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, aux_params)

        self._params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec.copy_params_from(self._arg_params, self._aux_params)
  
    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self._grad_req = grad_req

        if not for_training:
            assert not inputs_need_grad

        assert shared_module is None, 'Shared module is not supported.'

        self._data_desc = \
            [x if isinstance(x, DataDesc) else DataDesc(*x) for x in data_shapes]
        if label_shapes is not None:
            self._label_desc = \
                [x if isinstance(x, DataDesc) else DataDesc(*x) for x in label_shapes]
        else:
            self._label_desc = None

        # Bind and create executor
        # 1. Shapes/types
        input_names = self._symbol.list_arguments()

        feed_shapes = {}
        feed_shapes.update({desc.name : desc.shape for desc in self._data_desc})
        feed_shapes.update({desc.name : desc.shape for desc in self._label_desc})
        feed_dtypes = {k: mx_real_t for k in input_names}

        arg_shapes, out_shapes, aux_shapes = self._symbol.infer_shape(**feed_shapes)
        arg_dtypes, out_dtypes, aux_dtypes = self._symbol.infer_type(**feed_dtypes)

        print(arg_shapes, out_shapes, aux_shapes)

        # 2. Create arrays for binding
        arg_params = [nd.empty(shape, self._default_context, dtype=dtype)
                      for shape, dtype in zip(arg_shapes, arg_dtypes)]
        aux_params = [nd.empty(shape, self._default_context, dtype=dtype)
                      for shape, dtype in zip(aux_shapes, aux_dtypes)]
        grad_dict = {name : nd.empty(shape, self._default_context, dtype=dtype)
                     for name, shape, dtype in zip(input_names, arg_shapes, arg_dtypes)
                     if name in self._param_names}
        print('Grad names: ', grad_dict.keys())

        # 3. Create group2ctx
        group2ctx = {'group:%d' % i : self._context[i] for i in range(len(self._context))}

        # 4. Bind
        self._exec = self._symbol.bind(self._default_context,
                                       args=arg_params,
                                       args_grad=grad_dict,
                                       grad_req=self._grad_req,
                                       aux_states=aux_params,
                                       group2ctx=group2ctx)

        self._binded = True 

    def reshape(self, data_shapes, label_shapes=None):
        """Reshape the module for new input shapes.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        """
        # TODO
        raise NotImplementedError()

    def init_optimizer(self, kvstore=None, optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert kvstore is None, 'KVStore is not supported'
        assert self._binded and self._params_initialized
        if self._optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        if isinstance(optimizer, str):
            batch_size = self._exec.batch_size
            optimizer_params = dict(optimizer_params)
            idx2name = enumerate(self._param_names)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = 1.0/batch_size
            optimizer = opt.create(optimizer,
                                   sym=self._symbol, param_idx2name=idx2name,
                                   **optimizer_params)

        self._optimizer = optimizer
        self._updater = opt.get_updater(optimizer)

        self._optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None

    def borrow_optimizer(self, shared_module):
        raise NotImplementedError('Shared module is not supported.')

    def forward(self, data_batch, is_train=None):
        assert self._binded and self._params_initialized
        self._exec.forward(data_batch, is_train)

    def backward(self, out_grads=None):
        assert self._binded and self._params_initialized
        self._exec.backward(out_grads=out_grads)

    def update(self):
        assert self._binded and self._params_initialized and self._optimizer_initialized
        self._params_dirty = True
        _update_params(self._exec.arg_arrays,
                       self._exec.grad_arrays,
                       updater=self._updater,
                       num_device=1,  # Only update on cpu.
                       kvstore=None)

    def get_outputs(self, merge_multi_context=True):
        """Get outputs of the previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self._binded and self._params_initialized
        raise NotImplementedError()

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[grad1, grad2]`. Otherwise, it
        is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self._binded and self._params_initialized and self._inputs_need_grad
        raise NotImplementedError()

    def update_metric(self, eval_metric, labels):
        eval_metrix.update(labels, self._exec.outputs)

    def _sync_params_from_devices(self):
        pass

    def save_optimizer_states(self, fname):
        # TODO
        pass

    def load_optimizer_states(self, fname):
        # TODO
        pass

    def install_monitor(self, mon):
        # TODO
        pass
