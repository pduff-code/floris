import inspect

def show_params(fi, params=None, verbose=False, wake_velocity_model=True,
                    wake_deflection_model=True, turbulence_model=True):

    if wake_velocity_model is True:
        obj = 'fi.floris.farm.wake.velocity_model'
        props = get_props(obj, fi)

        if verbose == True:
            print('='.join(['=']*39))
        else:
            print('='.join(['=']*19))
        print('Wake Velocity Model Parameters:', \
               fi.floris.farm.wake.velocity_model.model_string, 'model')

        if params is not None:
            props_subset = get_props_subset(params, props)
            if verbose is False:
                print_props(obj, fi, props_subset)
            else:
                print_prop_docs(obj, fi, props_subset)

        else:
            if verbose is False:
                print_props(obj, fi, props)
            else:
                print_prop_docs(obj, fi, props)

    if wake_deflection_model is True:
        obj = 'fi.floris.farm.wake.deflection_model'
        props = get_props(obj, fi)

        if verbose == True:
            print('='.join(['=']*39))
        else:
            print('='.join(['=']*19))
        print('Wake Deflection Model Parameters:', \
              fi.floris.farm.wake.deflection_model.model_string, 'model')

        if params is not None:
            props_subset = get_props_subset(params, props)
            if props_subset: # true if the subset is not empty
                if verbose is False:
                    print_props(obj, fi, props_subset)
                else:
                    print_prop_docs(obj, fi, props_subset)

        else:
            if verbose is False:
                print_props(obj, fi, props)
            else:
                print_prop_docs(obj, fi, props)

    if turbulence_model is True:
        obj = 'fi.floris.farm.wake.turbulence_model'
        props = get_props(obj, fi)

        if verbose == True:
            print('='.join(['=']*39))
        else:
            print('='.join(['=']*19))
        print('Wake Turbulence Model Parameters:', \
              fi.floris.farm.wake.turbulence_model.model_string, 'model')

        if params is not None:
            props_subset = get_props_subset(params, props)
            if props_subset: # true if the subset is not empty
                if verbose is False:
                    print_props(obj, fi, props_subset)
                else:
                    print_prop_docs(obj, fi, props_subset)

        else:
            if verbose is False:
                print_props(obj, fi, props)
            else:
                print_prop_docs(obj, fi, props)

def get_params(fi, params=None, wake_velocity_model=True,
                    wake_deflection_model=True, turbulence_model=True):
    model_params = {}

    if wake_velocity_model is True:
        wake_vel_vals = {}
        obj = 'fi.floris.farm.wake.velocity_model'
        props = get_props(obj, fi)
        if params is not None:
            props_subset = get_props_subset(params, props)
            wake_vel_vals = get_prop_values(obj, fi, props_subset)
        else:
            wake_vel_vals = get_prop_values(obj, fi, props)
        model_params['Wake Velocity Parameters'] = wake_vel_vals

    if wake_deflection_model is True:
        wake_defl_vals = {}
        obj = 'fi.floris.farm.wake.deflection_model'
        props = get_props(obj, fi)
        if params is not None:
            props_subset = get_props_subset(params, props)
            wake_defl_vals = get_prop_values(obj, fi, props_subset)
        else:
            wake_defl_vals = get_prop_values(obj, fi, props)
        model_params['Wake Deflection Parameters'] = wake_defl_vals

    if turbulence_model is True:
        wake_defl_vals = {}
        obj = 'fi.floris.farm.wake.turbulence_model'
        props = get_props(obj, fi)
        if params is not None:
            props_subset = get_props_subset(params, props)
            wake_defl_vals = get_prop_values(obj, fi, props_subset)
        else:
            wake_defl_vals = get_prop_values(obj, fi, props)
        model_params['Wake Turbulence Parameters'] = wake_defl_vals

    return model_params

def set_params(fi, params, verbose=True):
    for param_dict in params:
        if param_dict == 'Wake Velocity Parameters':
            obj = 'fi.floris.farm.wake.velocity_model'
            props = get_props(obj, fi)
            for prop in params[param_dict]:
                if prop in [val[0] for val in props]:
                    exec(obj + '.' + prop + ' = ' + \
                         str(params[param_dict][prop]))
                    if verbose is True:
                        print('Wake velocity parameter ' + prop + ' set to ' + \
                            str(params[param_dict][prop]))
                else:
                    raise Exception(('Wake deflection parameter \'{}\' ' + \
                        'not part of current model. Value \'{}\' was not ' + \
                        'used.').format(prop, params[param_dict][prop]))

        if param_dict == 'Wake Deflection Parameters':
            obj = 'fi.floris.farm.wake.deflection_model'
            props = get_props(obj, fi)
            for prop in params[param_dict]:
                if prop in [val[0] for val in props]:
                    exec(obj + '.' + prop + ' = ' + \
                         str(params[param_dict][prop]))
                    if verbose is True:
                        print('Wake deflection parameter ' + prop + \
                              ' set to ' + str(params[param_dict][prop]))
                else:
                    raise Exception(('Wake deflection parameter \'{}\' ' + \
                        'not part of current model. Value \'{}\' was not ' + \
                        'used.').format(prop, params[param_dict][prop]))
        
        if param_dict == 'Wake Turbulence Parameters':
            obj = 'fi.floris.farm.wake.turbulence_model'
            props = get_props(obj, fi)
            for prop in params[param_dict]:
                if prop in [val[0] for val in props]:
                    exec(obj + '.' + prop + ' = ' + \
                         str(params[param_dict][prop]))
                    if verbose is True:
                        print('Wake turbulence parameter ' + prop + \
                              ' set to ' + str(params[param_dict][prop]))
                else:
                    raise Exception(('Wake turbulence parameter \'{}\' ' + \
                        'not part of current model. Value \'{}\' was not ' + \
                        'used.').format(prop, params[param_dict][prop]))

def get_props_subset(params, props):
    prop_names = [prop[0] for prop in props]
    try:
        props_subset_inds = [prop_names.index(param) \
                                for param in params]
    except:
        props_subset_inds = []
        print('Parameter(s)', ', '.join(params), 'does(do) not exist.')
    props_subset = [props[i] for i in props_subset_inds]
    return props_subset

def get_props(obj, fi):
    return inspect.getmembers(eval(obj + '.__class__'), \
                              lambda obj: isinstance(obj, property))

def get_prop_values(obj, fi, props):
    prop_val_dict = {}
    for val in props:
        prop_val_dict[val[0]] = eval(obj + '.' + val[0])
    return prop_val_dict

def print_props(obj, fi, props):
    print('-'.join(['-']*19))
    for val in props:
        print(val[0] + ' = ' + str(eval(obj + '.' + val[0])))
    print('-'.join(['-']*19))

def print_prop_docs(obj, fi, props):
    for val in props:
        print('-'.join(['-']*39) + '\n', val[0] + ' = ' + str(eval(obj + '.' \
            + val[0])), '\n', eval(obj + '.__class__.' + val[0] + '.__doc__'))
    print('-'.join(['-']*39))