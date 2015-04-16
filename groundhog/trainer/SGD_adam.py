import numpy
import time
import logging

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const

logger = logging.getLogger(__name__)


class SGD(object):

    def __init__(self,
                 model,
                 state,
                 data):
        """
        This class implements the Nesterov Momentum in the formulation at the below paper:
            http://arxiv.org/pdf/1212.0901v2.pdf
        With rmsprop as well.
        Parameters:
            :param model:
                Class describing the model used. It should provide the
                 computational graph to evaluate the model, and have a
                 similar structure to classes on the models folder
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
            :param rmsprop_decay:
                Decay parameter for rmsprop.
        """

        if "beta1" in state:
            self.beta1 = state["beta1"]
        else:
            self.beta1 = 0.1

        if "beta2" in state:
            self.beta2 = state["beta2"]
        else:
            self.beta2 = 0.001

        if "adaeps" not in state:
            state["adaeps"] = 1e-8

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))

        print("My state is: ")
        print(state)
        print("Adam!!!")

        _params = [p for p in model.params if p not in model.exclude_params]
        _param_grads = [g for p, g in zip(model.params, model.param_grads) if p not in model.exclude_params]

        def shared_clone(p, name):
            if str(p.__class__).find('cuda') >= 0:
                func = theano.shared
            else:
                print "+++"
                func = TT._shared
            return func(
                    numpy.zeros(p.get_value().shape,
                        dtype=theano.config.floatX),
                    name=p.name)

        self.gs = [shared_clone(p, name=p.name+'_g') for p in _params]
        self.rms_gs = [shared_clone(p, name="rms_%s" % p.name) for p in _params]
        self.mean_gs = [shared_clone(p, name="mean_gs_%s" % p.name) for p in _params]
        self.updirs = [shared_clone(p, name="updir_%s" % p.name) for p in _params]

        i = theano.shared(0., name="i")
        i_t = i + 1.0

        lr = TT.scalar('lr')
        self.eps = state['adaeps']
        fix1 = 1 - (1 - self.beta1)**i_t
        fix2 = 1 - (1 - self.beta2)**i_t
        lr_t = lr * (TT.sqrt(fix2) / fix1)

        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################
        logger.debug('Constructing grad function')
        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]

        clone_inputs = _param_grads + self.update_rules + \
                       self.prop_exprs + [model.train_cost]
        if model.lm_alpha:
            clone_inputs += [model.lm_alpha.out]
        else:
            clone_inputs += [theano.shared(numpy.zeros((1,)), theano.config.floatX)]
        rval = theano.clone(clone_inputs, replace=zip(model.inputs, loc_data))

        nparams = len(_params)

        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
                          for x, p in zip(gs, _params) if p not in self.model.exclude_params_for_norm))
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g, p in zip(gs, _params):

                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p, tmpg))
                else:
                    _gs.append(g)
            gs = _gs

        rms_gs = []
        mean_gs = []
        moment_gs = gs
        updirs = []

        store_gs = zip(self.gs, moment_gs)
        updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]
        for p, s, g, mg, rms in zip(_params, self.gs, gs, self.mean_gs, self.rms_gs):
            print "Using adam for param %s" % p.name
            mg_t = (1.0 - self.beta1) * mg + self.beta1 * g
            r_t = (1.0 - self.beta2) * rms + self.beta2 * g**2
            rms_t = TT.sqrt(r_t) + self.eps
            g_t = -mg_t / (rms_t)
            updirs.append(g_t)
            rms_gs.append(r_t)
            mean_gs.append(mg_t)

        store_rms = zip(self.rms_gs, rms_gs)
        store_mgs = zip(self.mean_gs, mean_gs)
        store_updirs = zip(self.updirs, updirs)
        updates += store_rms
        updates += store_mgs
        updates += store_updirs
        updates.append((i, i_t))

        print('Compiling grad function')
        st = time.time()

        self.train_fn = theano.function([], outs,
                                        name='train_function',
                                        updates = updates,
                                        givens = zip(model.inputs, loc_data))

        print 'took', time.time() - st

        self.lr = numpy.float32(state['lr'])
        new_params = [p + lr_t * ud[0] for p, ud in zip(_params, store_updirs)]
        p_updates = zip(_params, new_params)

        self.update_fn = theano.function([lr], [],
                                         name='update_function',
                                         allow_input_downcast=True,
                                         on_unused_input="warn",
                                         updates = p_updates)
        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                       ['cost',
                        'error',
                        'time_step',
                        'whole_time',
                        'lr',
                        'avg_alpha']

    def __call__(self):

        batch = self.data.next()
        assert batch

        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)

        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)

        # Run the training function
        g_st = time.time()
        rvals = self.train_fn()
        for schedule in self.schedules:
            schedule(self, rvals[-2])

        self.state['lr'] = float(self.lr)
        self.update_fn(self.state['lr'])

        g_ed = time.time()

        cost = rvals[-2]
        avg_alpha = numpy.mean(numpy.mean(rvals[-1], axis=0))
        if numpy.isnan(cost) or numpy.isinf(cost):
            raise Exception('Got NaN in the cost!')

        self.old_cost = cost
        whole_time = time.time() - self.step_timer

        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' step time %s whole time %s lr %.2e %f'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr), float(avg_alpha)]
            print msg % tuple(vals)

        self.step += 1

        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                    ('lr', float(self.lr)),
                    ('time_step', float(g_ed - g_st)),
                    ('whole_time', float(whole_time)),
                    ('avg_alpha', float(avg_alpha))] + zip(self.prop_names, rvals))
        return ret
