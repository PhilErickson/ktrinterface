""" Tests to use with Nosetests """
from __future__ import print_function, division
from nose.tools import *
import numpy as np
import pandas as pd
import numpy.random as rand
import statsmodels.api as sm
from scipy.stats import norm
from subprocess import call, Popen, PIPE
import ktrinterface.interface as ki
#import ktrinterface.knitro as ktr
#from ktrinterface.knitroNumPy import *
import tests.baseknitro as bk


#def setup():
#    """ Check installation """
#    call(["python", "setup.py", "install", "--user"])
#    import ktrinterface
#
#
#def teardown():
#    """ Remove package installation """
#    p = Popen(["pip", "uninstall", "ktrinterface"], stdin=PIPE, stderr=PIPE,
#              stdout=PIPE, universal_newlines=True)
#    p.stdin.write("y")


class TestBasic(object):
    """ Class to test a basic unconstrained optimization problem """
    def __init__(self):
        self.data = pd.DataFrame(np.array([1, 1]), columns=['first'])
        mydict = {'x': 2}

    def objective(self, coef):
        return abs(np.dot(self.data['first'],
                          np.array([norm.cdf(coef[0]), -.5])))

    def test_basic(self):
        """ Test basic solver process """
        result = ki.ktrsolve(fun=self.objective,
                               guess=1.0,
                               options={'outlev': 'none'})
        #result = ktr_int.solve()
        assert abs(result['coef']) < 1e-10


class TestMultBasic(object):
    """ Test basic unconstrained OLS """
    def __init__(self):
        self.data = self.gen_data()
    
    @staticmethod
    def gen_data():
        """ Generate OLS data """
        np.random.seed(1876)
        mu, sigma, n_sample = 0, 0.1, 1000
        x_data = rand.uniform(1, 60, n_sample)
        y_data = -18 + 2*x_data + rand.normal(mu, sigma, n_sample)
        data = pd.DataFrame(np.array([y_data, x_data]).T,
                            columns=['y', 'x'])
        data['cons'] = 1
        return data
    
    def objective(self, coef):
        """ OLS model """
        xvars = ['cons', 'x']
        xbeta = np.dot(self.data[xvars], coef[0:2])
        return -np.sum(norm.logpdf((self.data['y'] - xbeta), coef[2]))
    
    def test_mult(self):
        """ Test ktrsolve with OLS """
        guess = sm.OLS(self.data['y'], self.data[['cons', 'x']]).fit()
        sigma_guess = (guess.ssr / (guess.nobs - 2))**(1/2)
        guess = guess.params
        guess = np.hstack((guess, sigma_guess))
        result = ki.ktrsolve(fun=self.objective, guess=guess,
                             options={'outlev': 'none'})
        assert np.allclose(result['coef'],
                           np.array([[-18.05453956, 2.00006294, 0.04958287]]))


class TestBase(object):
    """ Base class for testing interface """
    def __init__(self):
        self.problem = {}
        self.ktr_defs = {}
        self.options = {}
        self.sparse = {}


class TestTobit(TestBase):
    """ Class to test performance with tobit model specification """
    def __init__(self):
        super(TestTobit, self).__init__()
        self.data = pd.DataFrame([])

    def data_gen(self):
        """ Generate basic tobit data """
        np.random.seed(1876)
        mu, sigma, n_sample = 0, 0.1, 1000
        x_data = rand.uniform(1, 60, n_sample)
        y_data = -18 + 2*x_data + rand.normal(mu, sigma, n_sample)
        y_data[y_data < 1] = 1
        y_data[y_data > 100] = 100
        data = pd.DataFrame(np.array([y_data, x_data]).T,
                            columns=['OverallRank', 'x'])
        data['cons'] = 1
        data['TopRanked'] = 1 * (data['OverallRank'] == 1)
        data['BottomRanked'] = 1 * (data['OverallRank'] == 100)
        data['InsideRanked'] = 1 * ((data['OverallRank'] > 1) &
                                    (data['OverallRank'] < 100))
        self.data = data

    def tobit_dens(self, xbeta, sigma):
        ''' Standard logged tobit probability density function
        Args:
            - xbeta: array of (x_i'*beta)
            - sigma: standard deviation
        Returns:
            density function evaluated at mu=xbeta
        '''
        first = np.array(self.data['TopRanked']) * \
                np.log(1 - norm.cdf(xbeta / sigma))
        #print("First:", first)
        second = np.array(self.data['BottomRanked']) * \
                 norm.logcdf(xbeta / sigma)
        #print("Second", second)
        third = np.array(self.data['InsideRanked']) * \
                (norm.logpdf((xbeta - self.data['OverallRank']) / sigma) -
                 np.log(sigma))
        #print("Third:", third)
        return first + second + third

    def xbeta_param_unpack(self, theta):
        """ Unpacker for parameters and generator for xbeta """
        beta = theta[:2]
        sigma = theta[2]
        x_vars = ['cons', 'x']
        xbeta = np.dot(self.data[x_vars], beta)
        return beta, sigma, xbeta

    def set_problem(self):
        """ Define objective """
        def tobit_logl(theta):
            """ Log likelihood for tobit density """
            beta, sigma, xbeta = self.xbeta_param_unpack(theta)
            tobit = self.tobit_dens(xbeta, sigma)
            return -np.sum(tobit)

        def constr(theta):
            """ Constraint vector binding objective away from NaN evals """
            beta, sigma, xbeta = self.xbeta_param_unpack(theta)
            alpha = norm.ppf(1 - 1e-15)
            return alpha*sigma - xbeta

        self.problem['objective'] = tobit_logl
        self.problem['constr'] = constr

    def set_ktr_defs(self):
        """ Set problem definitions from KNITRO Numpy example """
        m_const = self.data.shape[0]
        self.ktr_defs = {'bnds_lo': np.array([-ki.KTR_INFBOUND,
                                              -ki.KTR_INFBOUND,
                                              0]),
                         'c_type': np.array([ki.KTR_CONTYPE_LINEAR],
                                            np.int64).repeat(m_const),
                         'c_bnds_lo': np.zeros(m_const)}

    def set_options(self):
        """ Set problem options """
        self.options = {'outlev': 'none', 'debug': 1}

    def tobit_guess(self):
        """ Generate initial guess for tobit model """
        x_vars = ['cons', 'x']
        guess_beta = sm.OLS(self.data['OverallRank'],
                            self.data[x_vars]).fit()
        guess_beta = guess_beta.params
        guess_sigma = np.var(self.data['OverallRank']) * 10
        return np.hstack((guess_beta, guess_sigma))

    def test_tobit_mle(self):
        """ Test tobit density with ktrinterface.interface """
        self.data_gen()
        self.set_problem()
        guess = self.tobit_guess()
        self.set_ktr_defs()
        self.set_options()
        ktr_int = ki.Interface(fun=self.problem['objective'], guess=guess,
                               constr=self.problem['constr'],
                               ktr_defs=self.ktr_defs,
                               options=self.options)
        result = ktr_int.solve()
        assert True


class TestKtrEx(TestBase):
    """ Base class to test constrained optimization problem based on KNITRO
    Numpy example. Specifically, the problem is from the Hock & Schittkowski
    and is defined as

        min   100 (x2 - x1^2)^2 + (1 - x1)^2
        s.t.  x1 x2 >= 1
              x1 + x2^2 >= 0
              x1 <= 0.5

    which, given starting point (0, 1), should converge to (0.5, 2.0)
    """
    def __init__(self):
        super(TestKtrEx, self).__init__()

    def set_problem(self):
        """ Define optimization problem based on KNITRO Numpy example"""
        def objective(x):
            tmp = x[1] - x[0]*x[0]
            obj = 100.0 * tmp*tmp + (1.0 - x[0])*(1.0 - x[0])
            return obj
        def constr(x):
            return np.array([
                x[0] * x[1],
                x[0] + x[1]*x[1]
            ])
        def grad(x):
            tmp = x[1] - x[0]*x[0]
            return np.array([
                (-400.0 * tmp * x[0]) - (2.0 * (1.0 - x[0])),
                200.0 * tmp
            ])
        def jac(x):
            return np.array([
                x[1],
                x[0],
                1.0,
                2.0 * x[1]
            ])
        def hess(x, lambda_, sigma):
            return np.array([
                sigma * ((-400.0 * x[1]) + (1200.0 * x[0] * x[0]) + 2.0),
                (sigma * (-400.0 * x[0])) + lambda_[0],
                (sigma * 200.0) + (lambda_[1] * 2.0)
            ])
        self.problem['objective'] = objective
        self.problem['constr'] = constr
        self.problem['grad'] = grad
        self.problem['jac'] = jac
        self.problem['hess'] = hess

    def set_ktr_defs_full(self):
        """ Set problem definitions from KNITRO Numpy example """
        self.ktr_defs = {'obj_goal': ki.KTR_OBJGOAL_MINIMIZE,
                         'obj_type': ki.KTR_OBJTYPE_GENERAL,
                         'bnds_lo': np.array([-ki.KTR_INFBOUND,
                                              -ki.KTR_INFBOUND]),
                         'bnds_up': np.array([0.5, ki.KTR_INFBOUND]),
                         'c_type': np.array([ki.KTR_CONTYPE_QUADRATIC,
                                             ki.KTR_CONTYPE_QUADRATIC],
                                            np.int64),
                         'c_bnds_lo': np.array([1.0, 0.0]),
                         'c_bnds_up': np.array([ki.KTR_INFBOUND,
                                                ki.KTR_INFBOUND])}

    def set_sparse_full(self):
        """ Set sparcity structure for Jacobian and Hessian matrices """
        self.sparse['jac'] = np.ones((2, 2))
        self.sparse['hess'] = np.triu(np.ones((2, 2)))

class TestKtrExNoDeriv(TestKtrEx):
    """ Class to test problem with no defined gradient/hessian, no sparsity """
    def __init__(self):
        super(TestKtrExNoDeriv, self).__init__()

    def set_options(self):
        """ Set problem options """
        self.options = {'outlev': 'none', 'gradopt': 2, 'hessopt': 2,
                        'feastol': 1.0E-10}

    def test_constr_ktrcomp(self):
        """ Test constrained KNITRO example with no included gradient/hessian,
        comparing to the output from the example routine
        """
        self.set_problem()
        self.set_ktr_defs_full()
        self.set_options()
        baseline = bk.ktr_ex_no_deriv()
        #ktr_int = ki.Interface(fun=self.problem['objective'],
        #                       guess=np.array([0.0, 1.0]),
        #                       constr=self.problem['constr'],
        #                       ktr_defs=self.ktr_defs,
        #                       options=self.options)
        #result = ktr_int.solve()
        result = ki.ktrsolve(fun=self.problem['objective'],
                             guess=np.array([0.0, 1.0]),
                             constr=self.problem['constr'],
                             ktr_defs=self.ktr_defs,
                             options=self.options)
        assert np.allclose(result['coef'], baseline)

    def test_constr_raw(self):
        """ Test constrained KNITRO example with no included gradient/hessian,
        comparing to raw expected results
        """
        self.set_problem()
        self.set_ktr_defs_full()
        self.set_options()
        ktr_int = ki.Interface(fun=self.problem['objective'],
                               guess=np.array([0.0, 1.0]),
                               constr=self.problem['constr'],
                               ktr_defs=self.ktr_defs,
                               options=self.options)
        result = ktr_int.solve()
        assert np.allclose(result['coef'], np.array([0.5, 2.0]))

class TestKtrExDeriv(TestKtrEx):
    """ Class to test problem with defined gradient/hessian and sparsity """
    def __init__(self):
        super(TestKtrExDeriv, self).__init__()

    def set_options(self, hessian=False):
        """ Set KNITRO options from KNITRO Numpy example """
        self.options = {'outlev': 'none',
                        'hessopt': 1,
                        'hessian_no_f': 1,
                        'feastol': 1.0E-10}

    def test_constr_ktrcomp(self):
        """ Test constrained KNITRO example with included gradient/hessian,
        comparing to the output from the example routine
        """
        self.set_problem()
        self.set_ktr_defs_full()
        self.set_options()
        self.set_sparse_full()
        baseline = bk.ktr_ex_deriv()
        ktr_int = ki.Interface(fun=self.problem['objective'],
                               guess=np.array([0.0, 1.0]),
                               constr=self.problem['constr'],
                               grad=self.problem['grad'],
                               jac=self.problem['jac'],
                               hess=self.problem['hess'],
                               ktr_defs=self.ktr_defs,
                               options=self.options,
                               sparse_jac=self.sparse['jac'],
                               sparse_hess=self.sparse['hess'])
        result = ktr_int.solve()
        assert np.allclose(result['coef'], baseline)

    def test_constr_raw(self):
        """ Test constrained KNITRO example with included gradient/hessian but
        no explicit sparsity pattern, comparing to raw expected results
        """
        self.set_problem()
        self.set_ktr_defs_full()
        self.set_options()
        ktr_int = ki.Interface(fun=self.problem['objective'],
                               guess=np.array([0.0, 1.0]),
                               constr=self.problem['constr'],
                               grad=self.problem['grad'],
                               jac=self.problem['jac'],
                               hess=self.problem['hess'],
                               ktr_defs=self.ktr_defs,
                               options=self.options)
        result = ktr_int.solve()
        assert np.allclose(result['coef'], np.array([0.5, 2.0]))

