# For each curve, set 'refletivity' as [-1, +1] value
# back = panel == 0
# panel[ back ] = panel.mean()
# panel = normalize( panel )
# panel[ back ] = 0


# Fill the panel with the curves
"""y_pos = -(0.3*panel_side_base)
while y_pos < int(1.3 * panel_side_base): 
    x, y = funcs( func_name )
    # x, y = scale(x, y, 25, (0, panel_side_base))
    
    plt.plot(x, y + y_pos, linewidth=1, color=str(np.random.uniform(.5, 1))) # 
    y_pos += random.randint(15, 15)

# plt.show()

# Export the data and convert to Gray
panel = MatplotlibUtil.fig2data( plt.gcf() )
panel = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)

plt.clf()
if verbose:
    print( panel.shape, panel.min(), panel.max() )
    
# Return negative panel normalized between 0 and 1
return (255 - panel) / 255."""
    