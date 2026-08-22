print("Downloading the package labreport...")

# nacitanie kniznic
import numpy as np
import pandas as pd

import metrolopy as uc
import sigfig as sf
import sys
import os
import tempfile
import zipfile
import urllib.request
import warnings
import lmfit

# Disable DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # Also helpful for some packages
warnings.filterwarnings("ignore", category=SyntaxWarning)  # Do not show No syntax problems

def np_zip(list1, list2):
    return np.column_stack((list1, list2))

def sum_of_squares(arr1, arr2):
    return np.sum(np.square(np.ravel(arr1) - np.ravel(arr2)))

from IPython.display import YouTubeVideo
from numpy import array as v
sv = lambda zoznam: vector(zoznam)
from numpy import float64 as dc
from scipy.stats import sem as stsem
from numpy import mean as npmean
from numpy import std as npstd
from IPython.display import IFrame

std = lambda x: npstd(x,ddof=1).item()
mean = lambda x: npmean(x).item()
sem = lambda x: stsem(x).item()

def import_github_package(github_repo_url, module_name, branch='master'):
    """
    Downloads a GitHub repository as a zip file, extracts it,
    adds it to sys.path, and imports the specified module.

    Parameters:
      github_repo_url (str): The URL of the GitHub repository.
      module_name (str): The name of the package/module to import.
      branch (str): The branch to download (default is 'master').

    Returns:
      module: The imported module.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Construct the zip file URL (assumes the repository is public)
    zip_url = f"{github_repo_url}/archive/refs/heads/{branch}.zip"
    zip_path = os.path.join(temp_dir, f"{module_name.lower()}.zip")
    urllib.request.urlretrieve(zip_url, zip_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # The extracted folder usually has a name like "{module_name}-{branch}"
    extracted_dir = None
    for item in os.listdir(temp_dir):
        if os.path.isdir(os.path.join(temp_dir, item)) and item.startswith(module_name + "-"):
            extracted_dir = os.path.join(temp_dir, item)
            break

    if extracted_dir is None:
        raise Exception("Failed to locate the extracted package directory.")

    # Add the extracted directory to sys.path
    sys.path.insert(0, extracted_dir)

    # Import and return the module
    imported_module = __import__(module_name.lower())
    return imported_module

# Sigfig package
# repo = "https://github.com/drakegroup/sigfig" 
# module = "sigfig"
# package = import_github_package(repo, module)
# import sigfig as sf

# Metrolopy package
# repo = "https://github.com/nrc-cnrc/MetroloPy" 
# module = "MetroloPy"
# package = import_github_package(repo, module)
#import metrolopy as uc

uc.gummy.style = '+-'
from functools import wraps
from sigfig import round as sigfig_round

@wraps(sigfig_round)
def sfround(*args, **kwargs):
    kwargs.setdefault("cutoff", 99)
    return sigfig_round(*args, **kwargs)

def budget(gvel, gnames, notation='', transpose=True):

    # ---------------------------------------------------------
    # Helper for final form:
    # 2 significant figures, preserving trailing zeros
    # ---------------------------------------------------------

    def sf2str(x):
        x = float(x)

        if x == 0:
            return "0.0"

        # Rounding itself is done by sfround
        y = float(sfround(x, sigfigs=2))

        exponent = int(np.floor(np.log10(abs(y))))

        # Scientific notation for very small / large values
        if exponent <= -5 or exponent >= 5:
            return f"{y:.1e}"

        # Number of decimal places needed for 2 significant figures
        decimals = 1 - exponent

        if decimals > 0:
            return f"{y:.{decimals}f}"
        elif decimals == 0:
            return f"{y:.1f}"
        else:
            return f"{y:.0f}"


    # ---------------------------------------------------------
    # Temporary names
    #
    # A = indirect quantity
    # B, C, D, ... = direct quantities
    # ---------------------------------------------------------

    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if len(gnames) > len(letters):
        raise ValueError(
            "Too many quantities for one-letter auxiliary names."
        )

    aux_names = letters[:len(gnames)]

    indirect = aux_names[0]       # A
    direct = aux_names[1:]        # B, C, D, ...

    # temporary name -> original name
    name_map = dict(zip(aux_names, gnames))


    # ---------------------------------------------------------
    # Initial budget calculation
    # ---------------------------------------------------------

    table = gvel[0].budget(
        gvel[1:],
        xnames=direct
    )

    Unit = gvel[0].budget(
        gvel[1:],
        xnames=direct,
        uunit='%'
    ).df['Unit']


    # ---------------------------------------------------------
    # Create uncertainty-budget table
    # ---------------------------------------------------------

    db = table.df.astype(float, errors='ignore')

    if 'Unit' not in db.columns:
        db.insert(1, 'Unit', Unit)


    # ---------------------------------------------------------
    # Identify indirect quantity and temporarily rename it A
    # ---------------------------------------------------------

    components = list(db['Component'])

    indirect_rows = [
        name for name in components
        if name not in direct
    ]

    if len(indirect_rows) != 1:
        raise ValueError(
            f"Cannot identify indirect quantity. "
            f"Components: {components}"
        )

    indirect_current = indirect_rows[0]

    db['Component'] = db['Component'].replace(
        indirect_current,
        indirect
    )


    # ---------------------------------------------------------
    # Calculations with A, B, C, ...
    # ---------------------------------------------------------

    db.set_index('Component', inplace=True)

    db = db.reindex(direct + [indirect])

    db.drop(columns='s', inplace=True)

    dydx = db['|dy/dx|']
    db.drop(columns='|dy/dx|', inplace=True)

    db['rel. u %'] = (
        db['u'] / db['Value'] * 100
    )

    db['|dy/dx|'] = dydx

    db['|dy/dx|.u'] = (
        db['u'] * db['|dy/dx|']
    )

    db['vars'] = (
        db['u'] * db['|dy/dx|']
    )**2

    db.loc[indirect, 'vars'] = db['vars'].sum()

    db.loc[indirect, '|dy/dx|'] = np.float64(1)

    db.loc[indirect, 'u'] = np.sqrt(
        db.loc[indirect, 'vars']
    )

    db.loc[indirect, 'rel. u %'] = (
        db.loc[indirect, 'u']
        / db.loc[indirect, 'Value']
        * 100
    )

    db.loc[indirect, '|dy/dx|.u'] = (
        db.loc[indirect, 'u']
    )

    db['rel. vars %'] = (
        db['vars']
        / db.loc[indirect, 'u']**2
        * 100
    )

    db['s'] = (
        np.sqrt(db['rel. vars %']) / 10
    )


    # ---------------------------------------------------------
    # Restore original quantity names
    # ---------------------------------------------------------

    db.rename(
        index=name_map,
        inplace=True
    )

    db.set_index(
        'Unit',
        append=True,
        inplace=True
    )


    # =========================================================
    # notation = ''
    #
    # FINAL FORM
    #
    # Value and u:
    #   sfround(value, uncertainty=u)
    #
    # Everything else:
    #   sfround(x, sigfigs=2)
    # =========================================================

    if notation == '':

        table = db.copy().astype(object)

        # Value and uncertainty rounded together
        for idx in db.index:

            result = sfround(
                float(db.at[idx, 'Value']),
                uncertainty=float(db.at[idx, 'u'])
            )

            value_str, u_str = [
                x.strip()
                for x in result.split('±')
            ]

            table.at[idx, 'Value'] = value_str
            table.at[idx, 'u'] = u_str

        # Remaining quantities: 2 significant figures
        for col in db.columns:

            if col not in ['Value', 'u']:

                table[col] = db[col].map(
                    lambda x:
                        sf2str(x)
                        if pd.notna(x)
                        else ''
                )

        table = table.fillna('')


    # =========================================================
    # notation = 'decimal'
    #
    # Full precision in ordinary decimal notation
    # =========================================================

    elif notation == 'decimal':

        table = db.map(
            lambda x:
                np.format_float_positional(
                    float(x),
                    unique=True,
                    trim='-'
                )
                if pd.notna(x)
                else ''
        )


    # =========================================================
    # notation = 'scientific'
    #
    # Full precision in scientific notation
    # =========================================================

    elif notation == 'scientific':

        table = db.map(
            lambda x:
                np.format_float_scientific(
                    float(x),
                    unique=True,
                    trim='-'
                )
                if pd.notna(x)
                else ''
        )


    else:

        raise ValueError(
            "notation must be '', 'decimal', or 'scientific'"
        )


    # ---------------------------------------------------------
    # Transpose
    # ---------------------------------------------------------

    if transpose:
        table = table.transpose()


    return table

def ipyurl(url, storage='google', medium = 'image'):
    if medium == 'image':
        if storage=='google':
            url = url.replace('file/d/','uc?id=').replace('/view?usp=sharing', '')
        else:
            url = url.replace('?dl=0','?raw=1')
    if medium == 'table':
            url = url.replace('edit','preview')
    return url

def frame_wrapping(df, fillna=True):
    def format_line_break(value):
        if isinstance(value, str):
            value = value.replace('\n', '<br>')
        return value
    
    if fillna:
        # Replace column names containing 'Unnamed' with empty strings
        df.columns = [col if 'Unnamed' not in col else '' for col in df.columns]
    
        # Replace NaN values with empty strings
        df = df.fillna('')
    
    wrapped_df  = df.style.format(formatter=format_line_break)
    return wrapped_df 

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# from googletrans import Translator
# translator = Translator()
# trENGSK = lambda text:translator.translate(text, src='en', dest='sk').text.replace('.','. ')

# from molmass import Formula as vzorec
# from mendeleev import element as prvok
# from scipy.constants import R as Rv, zero_Celsius as Z0, N_A as NA
# from scipy.constants import find, physical_constants as konst

print('The contents of the package have been loaded successfully.')

def showURL(url, ht=424):

    '''shortcut for IFrame displaying various media at given url address;
       for interactive SageMath worksheets it is appropriate height 424 and width 100%
    '''
    from IPython.display import IFrame
    return IFrame(url, width='100%', height=ht)

def read_google_table(url):
    """
    Read data from a Google Sheets URL and return it as a pandas DataFrame.

    Parameters:
    - url (str): The URL of the Google Sheets document.
                 The URL should be in one of the following formats:
                 - 'https://docs.google.com/spreadsheets/d/{id}/edit#gid={gid}' (copied from address bar, document must be in a publicly accessible folder)
                 - 'https://docs.google.com/spreadsheets/d/{id}/edit?gid={gid}...' (copied from address bar, document must be in a publicly accessible folder)
                 - 'https://docs.google.com/spreadsheets/d/{id}/preview#gid={gid}'
                 - 'https://docs.google.com/spreadsheets/d/{id}/edit?usp=sharing' (obtained through the Share option)

    Returns:
    - pandas.DataFrame: A DataFrame containing the data from the Google Sheets document.
                        If the URL is not valid, returns an error message string.

    Example usage:
    >>> url = 'https://docs.google.com/spreadsheets/d/1WF2bZoDZhYQWek2tSNjB-Lg6EeD78sUfTENOdCGJyIA/edit#gid=1118323065'
    >>> df = read_google_table(url)
    >>> print(df.head())

    Note:
    - The Google Sheets document must be publicly accessible.
    - The function assumes that the data in the Google Sheets document is in a format compatible with CSV export.
    """
    if 'edit#' in url:
        URL = url.replace('edit#', 'export?format=csv&')
    elif 'edit?' in url:
        URL = url.replace('edit?', 'export?format=csv&')
    elif 'preview?' in url:
        URL = url.replace('preview?', 'export?format=csv&')
    elif 'preview#' in url:
        URL = url.replace('preview#', 'export?format=csv&')
    elif 'edit?usp=sharing' in url:
        URL = url.replace('edit?usp=sharing', 'export?format=csv')
    else:
        return "Not valid URL, see 'help(read_google_table)'."
    gdf = pd.read_csv(URL)
    gdf = gdf.replace('\n',' ', regex=True)
    gdf.columns = [col.replace('\n', ' ') for col in gdf.columns]
    return gdf

# ============================================================
# lmfit tools for SageMath
# Explicit symbolic models and first-order ODE models
# Always return full result dictionary
# Automatically print available dictionary keys
# Includes approximate confidence and prediction bands for ODE fits
# The ODE model information is stored inside fit["model_info"],
# so bands can be computed simply by ode_confidence_band(fit).
# ============================================================

import lmfit
import numpy as np


# ============================================================
# Internal helper: print available keys
# ============================================================

def _print_fit_keys(result):
    """
    Print available keys in the fit result dictionary.
    """
    print("Available fit dictionary keys:")
    for key in result.keys():
        print("  fit['" + key + "']")


# ============================================================
# Internal helper: add lmfit parameters
# ============================================================

def _add_lmfit_parameter(params, var_varfit, par_data):
    """
    Add one parameter to lmfit.Parameters.

    Accepted forms:

        (param, min_val, max_val, init_val, vary)

    or

        (param, min_val, max_val, vary)

    If init_val is not given, the midpoint of finite bounds is used.
    """
    if len(par_data) not in [4, 5]:
        raise ValueError(
            "Parameter data must have the form "
            "(param, min_val, max_val, vary) or "
            "(param, min_val, max_val, init_val, vary)."
        )

    par = par_data[0]
    name = str(par) + '_fit'
    var_varfit[par] = name

    par_min = float(par_data[1])
    par_max = float(par_data[2])
    par_vary = bool(par_data[-1])

    if len(par_data) == 5:
        par_init = float(par_data[3])
    else:
        if np.isfinite(par_min) and np.isfinite(par_max):
            par_init = (par_min + par_max) / 2
        elif np.isfinite(par_min):
            par_init = par_min + 1.0
        elif np.isfinite(par_max):
            par_init = par_max - 1.0
        else:
            par_init = 1.0

    params.add(
        name,
        min=par_min,
        max=par_max,
        value=par_init,
        vary=par_vary
    )


# ============================================================
# Basic fitting of explicit symbolic functions
# ============================================================

def f_eval_list(f, l):
    """
    Evaluate a one-variable Sage symbolic expression f on a list/array l.
    """
    f = SR(f)
    ivar = f.variables()[0]
    return np.array([f.subs(ivar == i) for i in l])


def fit_model_fun(f, data, params, var_varfit):
    """
    Evaluate an explicit symbolic model f after substituting fitted parameters.
    """
    f = SR(f)

    vars_list = list(var_varfit.keys())
    pd = params.valuesdict()

    subs_vals = [v == pd[var_varfit[v]] for v in vars_list]
    subs_vals_f = f.subs(subs_vals)

    vals = f_eval_list(subs_vals_f, data[:, 0])
    return vals


# ============================================================
# Helper: collect all useful lmfit outputs
# ============================================================

def _collect_lmfit_output(estim, var_varfit, data):
    """
    Collect useful outputs from lmfit.MinimizerResult.

    In these wrappers the residual is defined as

        residual = model_values - data_values

    Therefore

        best_fit = data_values + residual
    """
    t_values = np.array(data[:, 0], dtype=float).flatten()
    y_data = np.array(data[:, 1], dtype=float).flatten()

    residuals = np.array(estim.residual, dtype=float).flatten()

    # Since residual = model - data
    best_fit = y_data + residuals

    # Fitted parameter values
    fit_pars_dict = {
        v: estim.params[var_varfit[v]].value
        for v in var_varfit.keys()
    }

    # Standard errors of fitted parameters
    stderr_dict = {
        v: estim.params[var_varfit[v]].stderr
        for v in var_varfit.keys()
    }

    # Relative standard errors in percent
    rel_stderr_dict = {}

    for v in var_varfit.keys():
        lmfit_name = var_varfit[v]
        par = estim.params[lmfit_name]

        value = par.value
        stderr = par.stderr

        if stderr is None or value == 0:
            rel_stderr_dict[v] = None
        else:
            rel_stderr_dict[v] = abs(stderr / value) * 100

    # Detailed parameter summary
    param_summary = {}

    for v in var_varfit.keys():
        lmfit_name = var_varfit[v]
        par = estim.params[lmfit_name]

        param_summary[v] = {
            "lmfit_name": lmfit_name,
            "value": par.value,
            "stderr": par.stderr,
            "relative_stderr_percent": rel_stderr_dict[v],
            "min": par.min,
            "max": par.max,
            "vary": par.vary,
            "init_value": par.init_value
        }

    # Varying parameters in the covariance-matrix order used by lmfit
    varying_lmfit_names = list(getattr(estim, "var_names", []))

    if len(varying_lmfit_names) == 0:
        varying_lmfit_names = [
            var_varfit[v]
            for v in var_varfit.keys()
            if estim.params[var_varfit[v]].vary
        ]

    lmfit_name_to_symbol = {
        var_varfit[v]: v
        for v in var_varfit.keys()
    }

    varying_symbols = [
        lmfit_name_to_symbol[name]
        for name in varying_lmfit_names
        if name in lmfit_name_to_symbol
    ]

    return {
        # Parameter information
        "params": fit_pars_dict,
        "stderr": stderr_dict,
        "relative_stderr_percent": rel_stderr_dict,
        "param_summary": param_summary,

        # Parameter ordering information
        "varying_params": varying_symbols,
        "varying_lmfit_names": varying_lmfit_names,

        # lmfit-style fitted values
        "best_fit": best_fit,

        # Alias
        "fit_values": best_fit,

        # Residuals
        "residuals": residuals,

        # Fit statistics
        "chisqr": estim.chisqr,
        "redchi": estim.redchi,
        "residual_standard_error": np.sqrt(estim.redchi),
        "aic": estim.aic,
        "bic": estim.bic,

        # Covariance matrix
        "covar": estim.covar,

        # Report and raw object
        "report": lmfit.fit_report(estim),
        "raw": estim,

        # Original data
        "t": t_values,
        "y_data": y_data,

        # Useful two-column arrays
        "data_fit": np.column_stack((t_values, best_fit)),
        "data_residuals": np.column_stack((t_values, residuals))
    }


# ============================================================
# Main lmfit wrapper for explicit symbolic functions
# Always returns full result dictionary
# ============================================================

def lmfit_fun(f, data, *params_data):
    """
    Fit an explicit one-variable symbolic function.

    Always returns a dictionary.
    """
    data = np.array(data, dtype=float)

    params = lmfit.Parameters()
    var_varfit = {}

    for par_data in params_data:
        _add_lmfit_parameter(params, var_varfit, par_data)

    fcn = lambda params: (
        fit_model_fun(f, data, params, var_varfit) - data[:, 1]
    ).astype(float)

    estim = lmfit.minimize(fcn, params, method='least_squares')

    result = _collect_lmfit_output(
        estim,
        var_varfit,
        data
    )

    _print_fit_keys(result)

    return result


# ============================================================
# Helper for refined ODE evaluation
# ============================================================

def refined_times_from_data(data_times, n=1):
    """
    Create time points for ODE integration.

    If n = 1:
        return original data times.

    If n > 1:
        insert n-1 intermediate points between consecutive data times.
    """
    data_times = [float(t) for t in data_times]

    if n == 1:
        return data_times

    times = []

    for i in range(len(data_times) - 1):
        t_left = data_times[i]
        t_right = data_times[i + 1]

        interval_times = np.linspace(t_left, t_right, n + 1)

        if i == 0:
            times.extend(interval_times)
        else:
            times.extend(interval_times[1:])

    return [float(t) for t in times]


# ============================================================
# ODE model evaluation
# ============================================================

def fit_model_1ode(model, dvars, ivar, ics, n, data, params, var_varfit):
    """
    Evaluate a system of first-order ODEs after substituting fitted parameters.

    One ODE example:

        model = [g]
        dvars = [v]
        ics = [v0]

    System example:

        model = [v, -g]
        dvars = [y, v]
        ics = [y0, v0]

    For one-dimensional ODEs this function internally adds a hidden
    dummy equation. Students can still write the natural model.
    """
    data = np.array(data, dtype=float)

    vars_list = list(var_varfit.keys())
    pd = params.valuesdict()

    subs_vals = [v == pd[var_varfit[v]] for v in vars_list]

    # SR(eq) makes the code robust also when some equation is plain 0.
    subs_vals_model = [SR(eq).subs(subs_vals) for eq in model]

    times = refined_times_from_data(data[:, 0], n)
    ics_num = [float(ic) for ic in ics]

    # ------------------------------------------------------------
    # Hidden dummy-variable trick for one-dimensional ODEs
    # ------------------------------------------------------------
    if len(dvars) == 1:
        dummy = SR.var('_lmfit_internal_dummy')

        augmented_model = [subs_vals_model[0], SR(0)]
        augmented_dvars = [dvars[0], dummy]
        augmented_ics = [ics_num[0], 0.0]

        sol = desolve_odeint(
            augmented_model,
            dvars=augmented_dvars,
            ivar=ivar,
            ics=augmented_ics,
            times=times
        )

        sol = np.array(sol, dtype=float)

        if n > 1:
            sol = sol[0::n, :]

        return sol[:, 0:1]

    # ------------------------------------------------------------
    # Genuine system of two or more first-order ODEs
    # ------------------------------------------------------------
    sol = desolve_odeint(
        subs_vals_model,
        dvars=dvars,
        ivar=ivar,
        ics=ics_num,
        times=times
    )

    sol = np.array(sol, dtype=float)

    if n > 1:
        sol = sol[0::n, :]

    return sol


# ============================================================
# Main lmfit wrapper for first-order ODE models
# Always returns full result dictionary
# Stores model information inside fit["model_info"]
# ============================================================

def lmfit_1ode(model, dvars, ivar, ics, n, data, fit_dvar, *params_data):
    """
    Fit a first-order ODE model using lmfit.

    Always returns a dictionary.

    The model specification is stored in

        fit["model_info"]

    so that confidence bands can be computed simply by

        band95 = ode_confidence_band(fit, sigma=2)
    """
    data = np.array(data, dtype=float)

    params = lmfit.Parameters()
    var_varfit = {}

    for par_data in params_data:
        _add_lmfit_parameter(params, var_varfit, par_data)

    column = dvars.index(fit_dvar)

    fcn = lambda params: (
        fit_model_1ode(
            model,
            dvars,
            ivar,
            ics,
            n,
            data,
            params,
            var_varfit
        )[:, column].flatten()
        - data[:, 1]
    ).astype(float)

    estim = lmfit.minimize(fcn, params, method='least_squares')

    result = _collect_lmfit_output(
        estim,
        var_varfit,
        data
    )

    # Store everything needed to recompute the ODE model later.
    result["model_info"] = {
        "model": model,
        "dvars": dvars,
        "ivar": ivar,
        "ics": ics,
        "n": n,
        "data": data,
        "fit_dvar": fit_dvar
    }

    _print_fit_keys(result)

    return result


# ============================================================
# Evaluate ODE model at chosen parameter values
# ============================================================

def ode_best_fit_at_params(fit, param_values):
    """
    Evaluate the ODE model stored in fit["model_info"] at chosen parameter values.

    Example:

        y_vals = ode_best_fit_at_params(fit, {g: 9.81})
    """
    if "model_info" not in fit:
        raise ValueError(
            "fit does not contain model_info. "
            "Re-run lmfit_1ode using the newest version of the package."
        )

    info = fit["model_info"]

    model = info["model"]
    dvars = info["dvars"]
    ivar = info["ivar"]
    ics = info["ics"]
    n = info["n"]
    data = info["data"]
    fit_dvar = info["fit_dvar"]

    data = np.array(data, dtype=float)

    params = lmfit.Parameters()
    var_varfit = {}

    for par, val in param_values.items():
        name = str(par) + '_fit'
        var_varfit[par] = name
        params.add(name, value=float(val), vary=False)

    column = dvars.index(fit_dvar)

    sol = fit_model_1ode(
        model,
        dvars,
        ivar,
        ics,
        n,
        data,
        params,
        var_varfit
    )

    return sol[:, column].flatten()


# ============================================================
# Approximate confidence band for ODE fit
# ============================================================

def ode_confidence_band(fit, sigma=2, rel_step=1e-6):
    """
    Approximate pointwise confidence band for an ODE fit.

    Usage:

        band95 = ode_confidence_band(fit, sigma=2)

    Mathematical idea:

        Var(y_hat(t_i)) approx J_i C J_i^T,

    where C is the covariance matrix of fitted parameters and J_i is
    the row of sensitivities dy(t_i)/dtheta_j.

    This is a pointwise approximate confidence band for the fitted mean model.
    It is not a full prediction interval for future measurements.
    """
    if "model_info" not in fit:
        raise ValueError(
            "fit does not contain model_info. "
            "Re-run lmfit_1ode using the newest version of the package."
        )

    info = fit["model_info"]
    data = np.array(info["data"], dtype=float)

    covar = fit["covar"]

    if covar is None:
        raise ValueError(
            "Covariance matrix is None. lmfit could not estimate parameter covariance."
        )

    covar = np.array(covar, dtype=float)

    params0 = fit["params"]

    if "varying_params" in fit:
        par_symbols = fit["varying_params"]
    else:
        # Fallback for older fit dictionaries
        all_pars = list(params0.keys())

        if covar.shape[0] == len(all_pars):
            par_symbols = all_pars
        else:
            par_symbols = [
                p for p in all_pars
                if ("stderr" in fit and fit["stderr"].get(p, None) is not None)
            ]

            if covar.shape[0] != len(par_symbols):
                raise ValueError(
                    "Cannot determine which parameters correspond "
                    "to the covariance matrix. Re-run lmfit_1ode."
                )

    if len(par_symbols) == 0:
        raise ValueError("No varying parameters found. Confidence band cannot be computed.")

    if covar.shape[0] != len(par_symbols):
        raise ValueError(
            "Covariance matrix size does not match the number of varying parameters."
        )

    y0 = np.array(fit["best_fit"], dtype=float).flatten()

    m = len(y0)
    p = len(par_symbols)

    J = np.zeros((m, p), dtype=float)

    for j, par in enumerate(par_symbols):
        val = float(params0[par])

        h = rel_step * max(abs(val), 1.0)

        params_plus = params0.copy()
        params_minus = params0.copy()

        params_plus[par] = val + h
        params_minus[par] = val - h

        y_plus = ode_best_fit_at_params(fit, params_plus)
        y_minus = ode_best_fit_at_params(fit, params_minus)

        J[:, j] = (y_plus - y_minus) / (2*h)

    # Pointwise variance:
    # Var(y_i) = J_i C J_i^T
    var_y = np.sum((J @ covar) * J, axis=1)

    # Numerical safety
    var_y = np.maximum(var_y, 0)

    se_fit = np.sqrt(var_y)

    lower = y0 - sigma * se_fit
    upper = y0 + sigma * se_fit

    return {
        "se_fit": se_fit,
        "lower": lower,
        "upper": upper,
        "data_lower": np.column_stack((data[:, 0], lower)),
        "data_upper": np.column_stack((data[:, 0], upper)),
        "data_band": np.column_stack((data[:, 0], lower, upper)),
        "jacobian": J,
        "used_parameters": par_symbols
    }


# ============================================================
# Approximate prediction band for ODE fit
# ============================================================

def ode_prediction_band(fit, sigma=2, rel_step=1e-6):
    """
    Approximate pointwise prediction band for future measured data.

    Usage:

        pred95 = ode_prediction_band(fit, sigma=2)

    This adds residual scatter to fitted-curve uncertainty:

        se_pred^2 = se_fit^2 + residual_standard_error^2
    """
    if "model_info" not in fit:
        raise ValueError(
            "fit does not contain model_info. "
            "Re-run lmfit_1ode using the newest version of the package."
        )

    info = fit["model_info"]
    data = np.array(info["data"], dtype=float)

    conf = ode_confidence_band(
        fit,
        sigma=1,
        rel_step=rel_step
    )

    y0 = np.array(fit["best_fit"], dtype=float).flatten()
    se_fit = conf["se_fit"]
    s_res = fit["residual_standard_error"]

    se_pred = np.sqrt(se_fit**2 + s_res**2)

    lower = y0 - sigma * se_pred
    upper = y0 + sigma * se_pred

    return {
        "se_pred": se_pred,
        "lower": lower,
        "upper": upper,
        "data_lower": np.column_stack((data[:, 0], lower)),
        "data_upper": np.column_stack((data[:, 0], upper)),
        "data_band": np.column_stack((data[:, 0], lower, upper)),
        "confidence_band": conf
    }


# ============================================================
# Sage plotting helper for bands
# ============================================================

def sage_band_plot(
    band,
    color='gray',
    fillcolor='gray',
    fillalpha=0.2,
    legend_label=None
):
    """
    Create a Sage plot of a confidence or prediction band.

    Input
    -----
    band :
        Dictionary returned by ode_confidence_band or ode_prediction_band.

    Returns
    -------
    Sage graphics object.
    """
    data_lower = band["data_lower"]
    data_upper = band["data_upper"]

    G_lower = spline(list(map(tuple, data_lower)))
    G_upper = spline(list(map(tuple, data_upper)))

    xmin = float(data_lower[0, 0])
    xmax = float(data_lower[-1, 0])

    G_band = plot(
        [G_lower, G_upper],
        fill={1: [0]},
        xmin=xmin,
        xmax=xmax,
        color=color,
        fillcolor=fillcolor,
        fillalpha=fillalpha
    )

    if legend_label is not None:
        x0 = xmin
        x1 = xmin + 0.001*(xmax - xmin)
        y0 = float(np.min(data_lower[:, 1])) - 10.0

        G_legend = line(
            [(x0, y0), (x1, y0)],
            color=color,
            legend_label=legend_label
        )

        return G_band + G_legend

    return G_band

# ============================================================
# Extended find_fit wrapper for labreport.sage
# ============================================================
#
# Requires already defined:
#
#     lmfit_fun
#     lmfit_1ode
#     ode_confidence_band
#     ode_prediction_band
#
# Main behavior:
#
#     find_fit(data, model)
#         -> original SageMath find_fit
#
#     find_fit(data, formula, (g, 5, 20, True))
#         -> lmfit_fun, compact dictionary {g: (estimate, stderr)}
#
#     find_fit(data, formula, (g, 5, 20, True), report='all')
#         -> lmfit_fun, full dictionary
#
#     model_ode = ode_model(...)
#
#     find_fit(data, model_ode)
#         -> lmfit_1ode, compact dictionary {g: (estimate, stderr)}
#
#     find_fit(data, model_ode, report='all')
#         -> lmfit_1ode, full dictionary
#
# ============================================================

from functools import wraps
import contextlib
import io
import numpy as np


# ------------------------------------------------------------
# Store original SageMath find_fit only once.
# This prevents accidental recursion if this cell is loaded repeatedly.
# ------------------------------------------------------------

if "_sage_find_fit_original" not in globals():
    _sage_find_fit_original = find_fit


# ------------------------------------------------------------
# Packaged ODE model constructor
# ------------------------------------------------------------

def ode_model(model, dvars, ivar, ics, fit_dvar, *params_data, n=1):
    """
    Create a packaged ODE model for the extended find_fit.

    Example
    -------

        model_ode = ode_model(
            model_y_odpor,
            [y, v],
            t,
            ics,
            y,
            (g, 8, 15, True),
            n=1
        )

        find_fit(data, model_ode)

        fit = find_fit(
            data,
            model_ode,
            report='all'
        )
    """
    return {
        "_lab_model_type": "ode",
        "model": model,
        "dvars": dvars,
        "ivar": ivar,
        "ics": ics,
        "fit_dvar": fit_dvar,
        "params_data": tuple(params_data),
        "n": n
    }


# ------------------------------------------------------------
# Internal helper: detect packaged ODE model
# ------------------------------------------------------------

def _is_packaged_ode_model(model):
    """
    Check whether model was created by ode_model(...).
    """
    return isinstance(model, dict) and model.get("_lab_model_type", None) == "ode"


# ------------------------------------------------------------
# Internal helper: detect lmfit parameter tuples
# ------------------------------------------------------------

def _looks_like_lmfit_params(args):
    """
    Check whether positional arguments look like lmfit parameter tuples:

        (param, min_val, max_val, vary)

    or

        (param, min_val, max_val, init_val, vary)
    """
    if len(args) == 0:
        return False

    return all(
        isinstance(a, (tuple, list)) and len(a) in [4, 5]
        for a in args
    )


# ------------------------------------------------------------
# Internal helper: initial guesses
# ------------------------------------------------------------

def _value_from_initial_guess(initial_guess, parameters, par, idx):
    """
    Extract initial value for one parameter.

    Supported forms:

        initial_guess = {a: 1, b: 2}

    or

        initial_guess = [1, 2]
    """
    if initial_guess is None:
        return None

    if isinstance(initial_guess, dict):
        return initial_guess.get(par, None)

    if isinstance(initial_guess, (list, tuple)):
        if idx < len(initial_guess):
            return initial_guess[idx]
        return None

    return None


# ------------------------------------------------------------
# Internal helper: parameter bounds
# ------------------------------------------------------------

def _bounds_for_parameter(bounds, parameters, par, idx):
    """
    Extract bounds for one parameter.

    Supported forms:

        bounds = {a: (-10, 10), b: (0, 5)}

    or

        bounds = [(-10, 10), (0, 5)]

    If bounds are not supplied, use unbounded lmfit parameters.
    """
    if bounds is None:
        return -np.inf, np.inf

    if isinstance(bounds, dict):
        return bounds.get(par, (-np.inf, np.inf))

    if isinstance(bounds, (list, tuple)):
        if idx < len(bounds):
            return bounds[idx]

    return -np.inf, np.inf


# ------------------------------------------------------------
# Internal helper: build lmfit parameter tuples
# ------------------------------------------------------------

def _make_lmfit_params_data(args, kwargs):
    """
    Construct lmfit-style parameter tuples.

    Priority
    --------

    1. Positional lmfit tuples:

            (g, 8, 15, True)
            (g, 8, 15, 9.8, True)

    2. Keyword fit_params:

            fit_params=[(g, 8, 15, True)]

    3. Sage-like parameters + optional initial_guess + optional bounds:

            parameters=[g]
            initial_guess=[9.8]
            bounds={g: (8, 15)}
    """

    # ------------------------------------------------------------
    # 1. Positional lmfit-style tuples
    # ------------------------------------------------------------
    if len(args) > 0:
        if _looks_like_lmfit_params(args):
            return tuple(args)

        raise ValueError(
            "For the lmfit extension, positional arguments after model "
            "must be parameter tuples such as "
            "(g, 8, 15, True) or (g, 8, 15, 9.8, True)."
        )

    # ------------------------------------------------------------
    # 2. Explicit fit_params keyword
    # ------------------------------------------------------------
    if "fit_params" in kwargs:
        fit_params = kwargs["fit_params"]

        if not isinstance(fit_params, (list, tuple)):
            raise TypeError("fit_params must be a list or tuple of parameter tuples.")

        return tuple(fit_params)

    # ------------------------------------------------------------
    # 3. Build from parameters, initial_guess, bounds
    # ------------------------------------------------------------
    parameters = kwargs.get("parameters", None)

    if parameters is None:
        raise ValueError(
            "For lmfit-based fitting, provide either lmfit-style "
            "parameter tuples, fit_params=[...], or parameters=[...]."
        )

    if not isinstance(parameters, (list, tuple)):
        parameters = [parameters]

    initial_guess = kwargs.get("initial_guess", None)
    bounds = kwargs.get("bounds", None)

    params_data = []

    for idx, par in enumerate(parameters):
        par_min, par_max = _bounds_for_parameter(bounds, parameters, par, idx)
        par_init = _value_from_initial_guess(initial_guess, parameters, par, idx)

        if par_init is None:
            params_data.append((par, par_min, par_max, True))
        else:
            params_data.append((par, par_min, par_max, par_init, True))

    return tuple(params_data)


# ------------------------------------------------------------
# Internal helper: decide whether formula call should use lmfit
# ------------------------------------------------------------

def _formula_call_requests_lmfit(args, kwargs, report_type):
    """
    Decide whether a formula-type call should be handled by lmfit_fun.

    Original Sage behavior is preserved for plain Sage calls such as:

        find_fit(data, model)
        find_fit(data, model, parameters=[a, b], variables=[x])

    lmfit is used when:
        - report='all'
        - positional lmfit-style parameter tuples are supplied
        - fit_params is supplied
    """
    if report_type == "all":
        return True

    if _looks_like_lmfit_params(args):
        return True

    if "fit_params" in kwargs:
        return True

    return False


# ------------------------------------------------------------
# Internal helper: available output string
# ------------------------------------------------------------

def _fit_outputs_string(fit):
    """
    Create a help string listing available keys in the fit dictionary.
    """
    lines = []
    lines.append("Available fit dictionary keys:")
    for key in fit.keys():
        lines.append("  fit['" + str(key) + "']")
    return "\n".join(lines)


def _attach_outputs_string(fit):
    """
    Add output/help strings into the fit dictionary.

    Both keys are provided:

        fit['output']
        fit['outputs']
    """
    fit["output"] = ""
    fit["outputs"] = ""

    output_text = _fit_outputs_string(fit)

    fit["output"] = output_text
    fit["outputs"] = output_text

    return fit


# ------------------------------------------------------------
# Internal helper: compact default output
# ------------------------------------------------------------

def _fit_result_to_default_output(fit, solution_dict=True, include_stderr=True):
    """
    Convert lmfit result dictionary to compact default output.

    Default:

        {g: (estimate, stderr)}

    If include_stderr=False:

        {g: estimate}

    If solution_dict=False and include_stderr=True:

        [(g, estimate, stderr)]

    If solution_dict=False and include_stderr=False:

        [g == estimate]
    """
    pars = fit["params"]
    stderrs = fit.get("stderr", {})

    if include_stderr:
        out = {
            p: (pars[p], stderrs.get(p, None))
            for p in pars.keys()
        }

        if solution_dict:
            return out

        return [
            (p, pars[p], stderrs.get(p, None))
            for p in pars.keys()
        ]

    else:
        if solution_dict:
            return dict(pars)

        return [p == val for p, val in pars.items()]


# ------------------------------------------------------------
# Internal helper: call lmfit function, suppress printing,
# and store output help string inside fit dictionary
# ------------------------------------------------------------

def _call_lmfit_and_attach_outputs(func, *args, **kwargs):
    """
    Call lmfit_fun or lmfit_1ode while suppressing internal printed output.

    The resulting dictionary receives:

        fit['output']
        fit['outputs']
    """
    with contextlib.redirect_stdout(io.StringIO()):
        fit = func(*args, **kwargs)

    if isinstance(fit, dict):
        fit = _attach_outputs_string(fit)

    return fit


# ------------------------------------------------------------
# Extended find_fit
# ------------------------------------------------------------

@wraps(_sage_find_fit_original)
def find_fit(data, model, *args, type='formula', report='default', **kwargs):
    """
    Extended SageMath find_fit.

    Default behavior is unchanged:

        find_fit(data, model)

    uses the original SageMath find_fit.

    Formula behavior:

        find_fit(data, model)
            -> original SageMath find_fit

        find_fit(data, model, (g, 5, 20, True))
            -> lmfit_fun, compact output {g: (estimate, stderr)}

        find_fit(data, model, (g, 5, 20, True), report='all')
            -> lmfit_fun, full dictionary

    ODE behavior:

        model_ode = ode_model(...)

        find_fit(data, model_ode)
            -> lmfit_1ode, compact output {g: (estimate, stderr)}

        find_fit(data, model_ode, report='all')
            -> lmfit_1ode, full dictionary

    Notes
    -----
    The keyword type='ode' is no longer needed for packaged ODE models.
    It is still accepted only for backward compatibility.
    """

    model_type = str(type).lower()
    report_type = str(report).lower()

    # Automatically detect packaged ODE model.
    if _is_packaged_ode_model(model):
        model_type = "ode"

    # ------------------------------------------------------------
    # 1. Formula model
    # ------------------------------------------------------------
    if model_type in ["formula", "sage", "standard", "fun", "function", "explicit"]:

        use_lmfit = _formula_call_requests_lmfit(args, kwargs, report_type)

        # Original SageMath behavior
        if not use_lmfit:
            return _sage_find_fit_original(data, model, *args, **kwargs)

        # lmfit-based formula fit
        params_data = _make_lmfit_params_data(args, kwargs)

        fit = _call_lmfit_and_attach_outputs(
            lmfit_fun,
            model,
            data,
            *params_data
        )

        if report_type == "all":
            return fit

        if report_type in ["default", "sage"]:
            solution_dict = kwargs.get("solution_dict", True)
            include_stderr = kwargs.get("include_stderr", True)

            return _fit_result_to_default_output(
                fit,
                solution_dict=solution_dict,
                include_stderr=include_stderr
            )

        raise ValueError(
            "Unknown report option. Use report='default' or report='all'."
        )

    # ------------------------------------------------------------
    # 2. ODE model fitted by lmfit_1ode
    # ------------------------------------------------------------
    if model_type in ["ode", "1ode", "odeint"]:

        # --------------------------------------------------------
        # Case A: model is packaged by ode_model(...)
        # --------------------------------------------------------
        if _is_packaged_ode_model(model):
            ode_info = model

            ode_rhs = ode_info["model"]
            dvars = ode_info["dvars"]
            ivar = ode_info["ivar"]
            ics = ode_info["ics"]
            fit_dvar = ode_info["fit_dvar"]
            n = ode_info.get("n", 1)

            packaged_params_data = tuple(ode_info.get("params_data", ()))

            # Explicit parameters in find_fit(...) override packaged parameters.
            if len(args) > 0 or "fit_params" in kwargs or "parameters" in kwargs:
                params_data = _make_lmfit_params_data(args, kwargs)
            else:
                params_data = packaged_params_data

            if len(params_data) == 0:
                raise ValueError(
                    "No fitted parameters were supplied. "
                    "Use ode_model(..., (g, 8, 15, True)) "
                    "or provide parameter tuples in find_fit(...)."
                )

        # --------------------------------------------------------
        # Case B: direct ODE syntax without ode_model(...)
        # Backward compatibility only.
        # --------------------------------------------------------
        else:
            required = ["dvars", "ivar", "ics", "fit_dvar"]

            for key in required:
                if key not in kwargs:
                    raise ValueError(
                        "For ODE fitting, preferably use ode_model(...). "
                        "For direct syntax, provide " + key + "."
                    )

            ode_rhs = model
            dvars = kwargs["dvars"]
            ivar = kwargs["ivar"]
            ics = kwargs["ics"]
            fit_dvar = kwargs["fit_dvar"]
            n = kwargs.get("n", 1)

            params_data = _make_lmfit_params_data(args, kwargs)

        fit = _call_lmfit_and_attach_outputs(
            lmfit_1ode,
            ode_rhs,
            dvars,
            ivar,
            ics,
            n,
            data,
            fit_dvar,
            *params_data
        )

        if report_type == "all":
            return fit

        if report_type in ["default", "sage"]:
            solution_dict = kwargs.get("solution_dict", True)
            include_stderr = kwargs.get("include_stderr", True)

            return _fit_result_to_default_output(
                fit,
                solution_dict=solution_dict,
                include_stderr=include_stderr
            )

        raise ValueError(
            "Unknown report option. Use report='default' or report='all'."
        )

    raise ValueError(
        "Unknown model type. Use an ordinary formula model or a model created by ode_model(...)."
    )


# ------------------------------------------------------------
# Extend documentation
# ------------------------------------------------------------

find_fit.__doc__ = (find_fit.__doc__ or "") + r"""

Extension in labreport.sage
---------------------------

Original SageMath behavior is preserved:

    find_fit(data, model)

calls the original SageMath find_fit.

Formula model examples
----------------------

1. Original SageMath behavior:

    find_fit(data, model)

2. Formula model with compact lmfit output:

    find_fit(
        data,
        model,
        (g, 5, 20, True)
    )

returns:

    {g: (estimate, stderr)}

3. Formula model with full lmfit output:

    fit = find_fit(
        data,
        model,
        (g, 5, 20, True),
        report='all'
    )

Then:

    print(fit["outputs"])
    fit["params"]
    fit["stderr"]
    fit["best_fit"]
    fit["chisqr"]
    print(fit["report"])

ODE model examples
------------------

4. ODE model packaged by ode_model(...):

    model_ode = ode_model(
        model_y_odpor,
        [y, v],
        t,
        ics,
        y,
        (g, 8, 15, True)
    )

    find_fit(data, model_ode)

returns:

    {g: (estimate, stderr)}

5. ODE model with full lmfit output:

    fit = find_fit(
        data,
        model_ode,
        report='all'
    )

Then:

    print(fit["outputs"])
    fit["params"]
    fit["stderr"]
    fit["best_fit"]
    fit["chisqr"]
    print(fit["report"])

For ODE fits with report='all':

    band95 = ode_confidence_band(fit, sigma=2)

To suppress standard errors in compact lmfit output:

    find_fit(data, model, (g, 5, 20, True), include_stderr=False)

or:

    find_fit(data, model_ode, include_stderr=False)
"""
