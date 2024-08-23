import joblib   
import logging
import dill as pickle

import numpy as np                  
import surmise
from surmise.emulation import emulator as sur_emu

from . import cachedir, lazydict
from .reader import data_list,observables
from .design import Design

class _Covariance:
    """
    Proxy object to extract observable sub-blocks from a covariance array.
    Returned by Emulator.predict().

    """
    def __init__(self, array, slices):
        self.array = array
        self._slices = slices

    def __getitem__(self, key):
        (obs1, subobs1), (obs2, subobs2) = key
        return self.array[
            ...,
            self._slices[obs1][subobs1],
            self._slices[obs2][subobs2]
        ]

class Emulator:
    def __init__(self, system):
        logging.info(
            'training emulator for system %s',
            system
        )
        self.system = system
        self.observables = observables
        self._slices = {}
        self.Y = {}  
        self.X = {}

        nobs = 0
        for obs, subobslist in self.observables:
            self._slices[obs] = {}
            self.Y[obs] = {}  
            self.X[obs] = {}
            for subobs in subobslist:
                y = []
                x = []
                y.append(data_list[system][obs][subobs]['Y'])
                x.append(data_list[system][obs][subobs]['x'])
                n = y[-1].shape[1]
                self._slices[obs][subobs] = slice(nobs, nobs + n)
                nobs += n
                self.Y[obs][subobs] = np.concatenate(y, axis=1).T
                self.X[obs][subobs] = np.concatenate(x, axis=0).reshape(-1, 1)

        self.design = Design(system)

        self.gps = [sur_emu(x=self.X[obs][subobs], theta=self.design.array, f=self.Y[obs][subobs], method='PCGP') 
                    for obs, subobslist in self.observables for subobs in subobslist]

    @classmethod
    def from_cache(cls, system, retrain=False, **kwargs):

        cachefile = cachedir / 'emulator' / '{}.pkl'.format(system)

        if not retrain and cachefile.exists():
            logging.debug('loading emulator for system %s from cache', system)
            emu = cls.__new__(cls)
            with open(cachefile, 'rb') as f:
                emu.__dict__ = pickle.load(f)
            return emu

        emu = cls(system, **kwargs)

        logging.info('writing cache file %s', cachefile)
        cachefile.parent.mkdir(exist_ok=True)
        with open(cachefile, 'wb') as f:
            pickle.dump(emu.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

        return emu

    def predict(self, Theta, return_cov=False):

        gp_mean_dict = {}
        gp_cov_dict = {}
        cov_blocks = []
        
        for obs, subobslist in self.observables:
            gp_mean_dict[obs] = {}
            gp_cov_dict[obs] = {}
            for subobs in subobslist:
                # Find the corresponding gp model
                gp_index = self.observables.index((obs, subobslist))
                gp = self.gps[gp_index]

                gp_pred = gp.predict(x=self.X[obs][subobs], theta=Theta)
                gp_mean_dict[obs][subobs] = gp_pred.mean().T
                gp_cov_dict[obs][subobs] = gp_pred.covx()
                cov_blocks.append(gp_pred.covx())
                cov_blocks.append(np.zeros_like(gp_pred.covx()))
        
        a = gp_cov_dict['R_AA']['C0']  # 70, 7, 7
        b = np.zeros_like(a)
        c = gp_cov_dict['R_AA']['C1']  # 70, 7, 7
        d = np.zeros_like(c)
    
        top = np.concatenate([a, b], axis=2) 
        bottom = np.concatenate([d, c], axis=2)  

        gp_cov_array = np.concatenate([top, bottom], axis=0)

        gp_cov_array = gp_cov_array.transpose(1, 0, 2) 
        if return_cov:
            return gp_mean_dict, _Covariance(gp_cov_array, self._slices)
        else:
            return gp_mean_dict
        
emulators = lazydict(Emulator.from_cache)

if __name__ == '__main__':
    import argparse
    from . import systems
    
    def arg_to_system(arg):
        if arg not in systems:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(
        description='train emulators for each collision system',
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--retrain', action='store_true',
        help='retrain even if emulator is cached'
    )
    parser.add_argument(
        'systems', nargs='*', type=arg_to_system,
        default=systems, metavar='SYSTEM',
        help='system(s) to train'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for s in kwargs.pop('systems'):
        emu = Emulator.from_cache(s, **kwargs)

        print(s)
        print('{} PCs explain {:.5f} of variance'.format(
            emu.npc,
            emu.pca.explained_variance_ratio_[:emu.npc].sum()
        ))

        for n, (evr, gp) in enumerate(zip(
                emu.pca.explained_variance_ratio_, emu.gps
        )):
            print(
                'GP {}: {:.5f} of variance, LML = {:.5g}, kernel: {}'
                .format(n, evr, gp.log_marginal_likelihood_value_, gp.kernel_)
            )