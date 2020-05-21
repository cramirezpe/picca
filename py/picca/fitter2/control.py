from __future__ import print_function, division
from . import parser, chi2
import numpy as np
import scipy as sp
import sys
from scipy.linalg import cholesky


class fitter2:
    '''
    Main interface for the fitter2. Creates cf models and runs the chi2.
    All of the functionality is single core.
    '''

    def __init__(self, chi2_file):
        ''' Read the config and initialize run settings '''
        self.dic_init = parser.parse_chi2(chi2_file)
        self.control = self.dic_init['control']
        self.run_chi2 = self.control.getboolean('chi2', False)
        self.replace_flag = self.control.getboolean('replace_fastMC', False)

        if self.replace_flag:
            if 'fast mc' not in self.dic_init:
                print('You asked to replace data with a mock, but there is no \'fast mc\' section.')
                sys.exit('ERROR: no fast mc section found')

            self.replace_fastMC()

        # Initialize the required objects
        if self.run_chi2:
            self.chi2 = chi2.chi2(self.dic_init)

    def run(self):
        ''' Run the fitter. This function only runs single core options '''

        if self.run_chi2:
            self.chi2.minimize()
            self.chi2.minos()
            self.chi2.chi2scan()
            self.chi2.fastMC()
            self.chi2.export()
        else:
            raise ValueError('You called "fitter.run()" without \
                asking for chi2. Set "chi2 = True" in [control]')

    def replace_fastMC(self):
        ''' Replace the data with a mock realization.
        Mainly used by the sampler, but can be used in other contexts as well.
        Warning: this also permanently rescales the covariance matrix if asked
        '''
        # We need to do one minimization to get the best fit parameters
        # This is not ideal, but other ways are even more convoluted
        # This will be done properly in fitter3
        chi2_local = chi2.chi2(self.dic_init)
        chi2_local.minimize()
        k = self.dic_init['fiducial']['k']
        pk_lin = self.dic_init['fiducial']['pk']
        pksb_lin = self.dic_init['fiducial']['pksb']

        sp.random.seed(self.dic_init['fast mc']['seed'])

        if 'covscaling' in self.dic_init['fast mc']:
            scale_fastmc = self.dic_init['fast mc']['covscaling']
        else:
            scale_fastmc = np.ones(len(self.dic_init['data sets']['data']))

        fidfast_mc = self.dic_init['fast mc']['fiducial']['values']
        fixfast_mc = self.dic_init['fast mc']['fiducial']['fix']
        # if set to true, will not add randomness to FastMC mock
        if 'forecast' in self.dic_init['fast mc']:
            forecast = self.dic_init['fast mc']['forecast']
        else:
            forecast = False

        for d, s in zip(self.dic_init['data sets']['data'], scale_fastmc):
            d.co = s*d.co
            d.ico = d.ico/s
            # no need to compute Cholesky when computing forecast
            if not forecast:
                d.cho = cholesky(d.co)

        fiducial_values = dict(chi2_local.best_fit.values).copy()
        for p in fidfast_mc:
            fiducial_values[p] = fidfast_mc[p]
            for d in self.dic_init['data sets']['data']:
                if p in d.par_names:
                    d.pars_init[p] = fidfast_mc[p]
                    d.par_fixed['fix_'+p] = fixfast_mc['fix_'+p]

        fiducial_values['SB'] = False
        for d in self.dic_init['data sets']['data']:
            d.fiducial_model = fiducial_values['bao_amp']*d.xi_model(k, pk_lin-pksb_lin, fiducial_values)

            fiducial_values['SB'] = True
            snl_per = fiducial_values['sigmaNL_per']
            snl_par = fiducial_values['sigmaNL_par']
            fiducial_values['sigmaNL_per'] = 0
            fiducial_values['sigmaNL_par'] = 0
            d.fiducial_model += d.xi_model(k, pksb_lin, fiducial_values)
            fiducial_values['SB'] = False
            fiducial_values['sigmaNL_per'] = snl_per
            fiducial_values['sigmaNL_par'] = snl_par
        del fiducial_values['SB']

        for d in self.dic_init['data sets']['data']:
            # if computing forecast, do not add randomness
            if forecast:
                d.da = d.fiducial_model
            else:
                g = sp.random.randn(len(d.da))
                d.da = d.cho.dot(g) + d.fiducial_model
            d.da_cut = d.da[d.mask]
