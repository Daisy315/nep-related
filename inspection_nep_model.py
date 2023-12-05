from glob import glob 
import numpy as np
from ase.io import read
from calorine.calculators import CPUNEP  
from calorine.nep import read_model
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import chebval
from pandas import DataFrame

model_fname = 'nep.txt'
model = read_model(model_fname)

### c^{ij}_{nk} for Radial descriptors ###
types = model.types
fig, axes = plt.subplots(figsize=(6, 5), nrows=len(types), ncols=len(types),
                         sharex=True, sharey=True, dpi=300)

for (s1, s2), data in model.radial_descriptor_weights.items():  ###s1和s2表示不同的元素种类
    irow, icol = types.index(s1), types.index(s2)
    ax = axes[irow][icol]
    for n, c_nk in enumerate(data):
        kwargs = dict(markersize=2, marker='o', alpha=0.5, label=f'{n}') 
        ax.plot(c_nk, **kwargs)
    ax.set_xticks([0, 4, 8])
    if irow == len(axes) - 1:
        ax.set_xlabel('$k$ index')
    if icol == 0:
        ax.set_ylabel('$c^{ij}_{nk}$')
    ax.text(0.05, 0.8, f'{s1}-{s2}', transform=ax.transAxes)
        
axes[0][-1].legend(title='$n$', loc='upper right', 
              bbox_to_anchor=(1.3, 1.08), frameon=False)

plt.subplots_adjust(hspace=0.0, wspace=0.0)  
fig.align_labels()
fig.savefig('./results/c_ij_nk-radial.png', dpi=300)

### visualize radial basis functions ###
def cutoff_func(rs, rcut):
    fc = 0.5 * (1 + np.cos(np.pi * rs / rcut))
    fc[rs > rcut] = 0
    return fc
    
types = model.types
rs = np.arange(0.5, model.radial_cutoff + 0.5, 0.01)
xs = 2 * (rs / model.radial_cutoff - 1) ** 2 - 1

fig, axes = plt.subplots(figsize=(6, 5), nrows=len(types), ncols=len(types),
                         sharex=True, sharey=True, dpi=300)

for (s1, s2), data in model.radial_descriptor_weights.items():
    irow, icol = types.index(s1), types.index(s2)
    ax = axes[irow][icol]

    for n, c_nk in enumerate(data):
        g_n = np.zeros(len(rs))
        for k in range(len(c_nk)):
            coeff = np.zeros((k + 1))
            coeff[-1] = 1
            f_k = 0.5 * (chebval(xs, coeff) + 1) * cutoff_func(rs, model.radial_cutoff)
            g_n += c_nk[k] * f_k
        kwargs = dict(alpha=0.5, label=n)
        ax.plot(rs, g_n, **kwargs)

    if irow == len(axes) - 1 and icol == 1:
        ax.set_xlabel('Interatomic distance $r$ (Å)')
    if irow == 1 and icol == 0:
        ax.set_ylabel('Radial basis function $g_n(r)$')
    ax.text(0.05, 0.8, f'{s1}-{s2}', transform=ax.transAxes)

axes[0][-1].legend(title='$n$', loc='upper right',
                   bbox_to_anchor=(1.3, 1.08), frameon=False)

plt.subplots_adjust(hspace=0.0, wspace=0.0)
fig.align_labels()
fig.savefig('./results/radial_basis_function.png', dpi=300)


### c^{ij}_{nk} for angular descriptors ###
types = model.types
fig, axes = plt.subplots(figsize=(6, 5), nrows=len(types), ncols=len(types),
                         sharex=True, sharey=True, dpi=300)

for (s1, s2), data in model.angular_descriptor_weights.items():  ###s1和s2表示不同的元素种类
    irow, icol = types.index(s1), types.index(s2)
    ax = axes[irow][icol]
    for n, c_nk in enumerate(data):
        kwargs = dict(markersize=2, marker='o', alpha=0.5, label=f'{n}') 
        ax.plot(c_nk, **kwargs)
    ax.set_xticks([0, 4, 8])
    if irow == len(axes) - 1:
        ax.set_xlabel('$k$ index')
    if icol == 0:
        ax.set_ylabel('$c^{ij}_{nk}$')
    ax.text(0.05, 0.8, f'{s1}-{s2}', transform=ax.transAxes)
        
axes[0][-1].legend(title='$n$', loc='upper right', 
              bbox_to_anchor=(1.3, 1.08), frameon=False)

plt.subplots_adjust(hspace=0.0, wspace=0.0)  
fig.align_labels()
fig.savefig('./results/c_ij_nk-angular.png', dpi=300)

### visualize angular basis functions ###
types = model.types
fig, axes = plt.subplots(figsize=(6, 5), nrows=len(types), ncols=len(types),
                         sharex=True, sharey=True, dpi=300)
rs = np.arange(0.5, model.angular_cutoff + 0.5, 0.01)
xs = 2 * (rs / model.angular_cutoff - 1) ** 2 - 1

for (s1, s2), data in model.angular_descriptor_weights.items():
    irow, icol = types.index(s1), types.index(s2)
    ax = axes[irow][icol]

    for n, c_nk in enumerate(data):
        g_n = np.zeros(len(rs))
        for k in range(len(c_nk)):
            coeff = np.zeros((k + 1))
            coeff[-1] = 1
            f_k = 0.5 * (chebval(xs, coeff) + 1) * cutoff_func(rs, model.angular_cutoff)
            g_n += c_nk[k] * f_k
        kwargs = dict(alpha=0.5, label=n)
        ax.plot(rs, g_n, **kwargs)

    if irow == len(axes) - 1 and icol == 1:
        ax.set_xlabel('Interatomic distance $r$ (Å)')
    if irow == 1 and icol == 0:
        ax.set_ylabel('Angular basis function $g_n(r)$')
    ax.text(0.05, 0.8, f'{s1}-{s2}', transform=ax.transAxes)

axes[0][-1].legend(title='$n$', loc='upper right',
                   bbox_to_anchor=(1.3, 1.08), frameon=False)

plt.subplots_adjust(hspace=0.0, wspace=0.0)
fig.align_labels()
fig.savefig('./results/angular_basis_function.png', dpi=300)

### Manipulating a model ### 
keys = [k for k in model.ann_parameters.keys() if k not in ['b1', 'b1_polar']]   #adjustment for NEP4
model_mod = read_model(model_fname)
for key in keys:

    for inner_key, value in model.ann_parameters[key].items():
        print(f'{inner_key:6} : {value.shape}')

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)  
    params = model.ann_parameters[key]['w0'].flatten() #plot the distribution of the weights w0 connecting the descriptor to the hidden layer on a logarithmic scale

    _ = ax.hist(np.log10(np.abs(params)), bins=100)  
    ax.set_xlabel('log$_{10}$($|w_{\\mu\\nu}^{(0)}|$)')
    ax.set_ylabel('Number of parameters')

    fig.align_labels()
    fig.savefig(f'./results/{key}_w0_uv.png', dpi=300)
    
    params = model.ann_parameters[key]['w0']
    print(f'number of non-zero parameters before pruning: {np.count_nonzero(params)}')
    params = np.where(np.log10(np.abs(params)) > -3, params, 0)       # remove the  parameters are actually rather small with absolute values  ≲10−3
    print(f'number of non-zero parameters after pruning: {np.count_nonzero(params)}')
    model_mod.ann_parameters[key]['w0'] = params
model_mod.write('./results/nep-modified.txt')

### Peeling the model, layer by layer ###
keys = [k for k in model.ann_parameters.keys() if k not in ['b1', 'b1_polar']]

for key in keys:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    params = model.ann_parameters
    nr = model.n_max_radial + 1
    na = model.n_max_angular + 1
    if model.l_max_4b != 0:
        model.l_max_4b = 1
    if model.l_max_5b != 0:
        model.l_max_5b = 1
    la = model.l_max_3b + model.l_max_4b + model.l_max_5b  
    acc_w_desc = np.mean(np.abs(params[key]['w0']), axis=0)
    small_values = (acc_w_desc < 0.1).sum()      #rate of low value 小于0.1 in des_component
    total = len(acc_w_desc)
    pct_small = 100 * small_values / total
    print(f"{pct_small:.2f}% {key}个w0的数值  < 0.1")
    for k, w in enumerate(acc_w_desc):
        kwargs = dict()
        label, color = '', 'C0'
        if k < nr:
            if k == 0:
                label = 'radial'
        else:
            color = 'C1'
            if k == nr:
                label = 'angular'
        kwargs = dict(color=color, label=label)
        ax.bar([k], [w], **kwargs)
    
    ax.set_xlabel(r'Descriptor component')
    ax.set_ylabel(r'Mean absolute weights')
    ax.legend(frameon=False)
    
    labels = [f'{r}' for r in range(nr)]
    labels += [f'{n},{l}' for l in range(la) for n in range(na)]
    ax.set_xticks(range(len(labels)), labels=labels)
    plt.xticks(rotation=90)
    fig.savefig(f'./results/{key}_des_component.png', dpi=300)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    w0 = params[key]['w0']
    w1 = params[key]['w1']
    b0 = params[key]['b0']
    xs = range(1, len(b0) + 1)
    acc_weights = np.mean(np.abs(w0), axis=1)
    ax.bar(xs, acc_weights,
        label=r'$N_{\mathrm{des}}^{-1} \sum_{\nu}^{N_{\mathrm{des}}} |w_{\mu\nu}^{(0)}|$')
    abs_b0 = np.abs(b0.flatten())
    ax.bar(xs, -np.abs(b0.flatten()), label=r'$|b_{\mu}^{(0)}|$ (bias)')
    ax.plot(xs, w1.flatten(), 'o-', label=r'$|w_{\mu}^{(1)}|$', c='green', markersize=4)
    ax.axhline(0, lw=1, c='k', alpha=0.5)
    ax.set_xlabel(r'Hidden neuron index $\mu$')
    ax.set_ylabel(r'Bias / Mean absolute weight')
    ax.legend(frameon=False)
    fig.savefig(f'./results/{key}_lat_component.png', dpi=300)
    abs_b0 = np.abs(b0.flatten())
    print('Approximate number of active neurons:', np.count_nonzero(abs_b0[abs_b0 > 0.01]))
    
### scaling parameters ###
types = model.types
fig, axes = plt.subplots(figsize=(6, 5), nrows=len(types), ncols=len(types),
                         sharex=True, sharey=True, dpi=300)

q_scaler = model.q_scaler

fig, ax = plt.subplots()
x = np.arange(len(q_scaler))
ax.bar(x, q_scaler)
ax.set_xlabel('des_component')
ax.set_ylabel('Scaling parameter')
ax.set_xticks(x)
ax.set_xticklabels(x) 
fig.savefig('./results/q_scaler.png')