# main script for testing boundary fill algorithms
from boundary_fill_simple import simple_boundary_fill, plot_layout_results

boundaries = [[0., 28000.], [8000., 0.], [11000., 0.], [21000., 22000.], [20000., 45000.], [0., 34000.]]
D = 240
wt_cap = 15
spacing = 7
x, y = simple_boundary_fill(boundaries, D, spacing)
print('This fill method yields ', len(x), ' turbines with ', spacing, ' D spacing' )
print('Total plant capacity = ', len(x)*15, ' MW')
plot_layout_results(boundaries, x, y, D)