import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from shapely.geometry import Point
from shapely.affinity import scale
from shapely.ops import unary_union
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection

class Plotter:
    """
    Initializes a Plotter object.

    Args:
    - config: a dictionary containing configuration options for plotting.
    """
    def __init__(self, config):
        """
        Initializes a Plotter object.

        Args:
        - config: a dictionary containing configuration options for plotting.
        """
        self.options = config['plotting']

    def set_ax_style(self, ax, title, legend=True):
        if self.options['map-extent'] is None:
            # Set the extent of the plot based on the range of the plotted points
            x_min, x_max, y_min, y_max = ax.get_extent()
            RANGE = np.max([y_max - y_min, x_max - x_min])
            y_mid = (y_max + y_min)/2
            x_mid = (x_max + x_min)/2
            ax.set_extent([x_mid - RANGE/2, x_mid + RANGE/2, y_mid - RANGE/2, y_mid + RANGE/2])
        else:
            ax.set_extent(self.options['map-extent'])

        if legend:
            ax.legend(fontsize=self.options['legend-fontsize'])

        # ax.gridlines(draw_labels=True, fontsize=self.options['tick-fontsize'])
        gridlines = ax.gridlines(draw_labels=True)
        gridlines.xlabel_style = {'size': self.options['tick-fontsize'], 'color': 'gray'}
        gridlines.ylabel_style = {'size': self.options['tick-fontsize'], 'color': 'gray'}
        
        ax.set_title(title, fontsize=self.options['title-fontsize'])

        # For the 'Longitude' label at the bottom
        ax.text(0.5, -0.08, 'Longitude (deg)', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=self.options['axis-fontsize'])

        # For the 'Latitude' label at the left
        ax.text(-0.12, 0.6, 'Latitude (deg)', va='center', ha='right',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=self.options['axis-fontsize'])

    def plot_route(self, state_vector_file, compressor, filename = None, ax = None, title = None):
        """
        Plots the route of a single aircraft based on its state vectors.

        Args:
        - state_vector_file: the file path containing state vectors for the aricraft.
        - compressor: a compressor object used to decode the state vectors.
        - filename(optional): the name of the file to save the plot to.
        - ax(optional): the axes to plot the data on.
        - title (optional): the title of the plot.
        """
        # Get the colormap to use for the plot
        colormap =  getattr(plt.cm, self.options['cmap'])

        if ax is not None and filename is not None:
            raise ValueError("Cannot specify both ax and filename")
        
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=[self.options['fig-size'][0], self.options['fig-size'][0]])
        else:
            fig = None
        # Add features to the plot
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linestyle=':')

        ys = np.zeros((2, self.options['point-precision']))
        df = compressor.decode_to_dataframe_from_file(state_vector_file)
        columns = ['lat', 'lon']
        df_interp = {col:[] for col in columns}
        df_interp = {'time':np.linspace(df['time'].iloc[0], df['time'].iloc[-1], num=self.options['point-precision'])}
        for col in columns:
            df_interp[col] = np.interp(df_interp['time'], df['time'], df[col])

        ys[0, :] = df_interp['lon']
        ys[1, :] = df_interp['lat']

        ax.plot(ys[0,:], ys[1,:], color = colormap(1.))
        if title is None:
            self.set_ax_style(ax, 'Aircraft Route', legend=False)
        else:
            self.set_ax_style(ax, title, legend=False)

        if fig is not None:
            fig.tight_layout()
        # Save the plot to a file
        if filename is not None:
            fig.savefig(filename)


    def plot_multiple_routes(self, state_vectors_files, compressor, filename = None, ax = None, title = None):
        """
        Plots the routes of multiple aircrafts based on their state vectors.

        Args:
        - state_vectors_files: a list of file paths containing state vectors for each aircraft.
        - compressor: a compressor object used to decode the state vectors.
        - filename(optional): the name of the file to save the plot to.
        - ax(optional): the axes to plot the data on.
        - title (optional): the title of the plot.
        """
        # Get the colormap to use for the plot
        colormap = getattr(plt.cm, self.options['cmap'])

        if ax is not None and filename is not None:
            raise ValueError("Cannot specify both ax and filename")
        
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=[self.options['fig-size'][0], self.options['fig-size'][0]])
        else:
            fig = None

        try:
            # Add features to the plot
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES, linestyle=':')
        except:
            raise ValueError("Invalid axes provided, did you remember to add \'subplot_kw={'projection': ccrs.PlateCarree()}\' to your axis?")
        ys = np.zeros((len(state_vectors_files), 2, self.options['point-precision']))
        # Loop through each state vectors file and plot the route
        for n, file in enumerate(state_vectors_files):
            # Decode the state vectors from the file and interpolate to get a fixed number of points
            df = compressor.decode_to_dataframe_from_file(file)
            columns = ['lat', 'lon']
            df_interp = {col:[] for col in columns}
            df_interp = {'time':np.linspace(df['time'].iloc[0], df['time'].iloc[-1], num=self.options['point-precision'])}
            for col in columns:
                df_interp[col] = np.interp(df_interp['time'], df['time'], df[col])

            ys[n, 0, :] = df_interp['lon']
            ys[n, 1, :] = df_interp['lat']
        mask = ~np.any(np.isnan(ys), axis=(1, 2))
        ys = ys[mask]

        if self.options['expectation-measure'] == 'mean':
            expectation_name = 'Mean Path'
            expectation = np.mean(ys, axis=0)
        elif self.options['expectation-measure'] == 'median':
            expectation_name = 'Median Path'
            expectation = np.median(ys, axis=0)
        elif self.options['expectation-measure'] == 'average':
            expectation_name = 'Average Path'
            expectation = np.average(ys, axis=0)
        else:
            raise ValueError(f"Expectation Measure not recognized: {self.options['expectation-measure']}")
        kde = gaussian_kde([np.ravel(ys[:,0,:]), np.ravel(ys[:,1,:])])
        kde_max = np.max(kde([np.ravel(ys[:,0,:]), np.ravel(ys[:,1,:])]))
        kde_min = np.min(kde([np.ravel(ys[:,0,:]), np.ravel(ys[:,1,:])]))
        segments = []
        colors = []

        colors_value = np.array([(kde([ys[i,0,:], ys[i,1,:]]) - kde_min)/(kde_max - kde_min) for i in range(ys.shape[0])])
        for i in range(ys.shape[0]):
            for j in range(ys.shape[2]-1):
                segment = [(ys[i,0,j], ys[i,1,j]), (ys[i,0,j+1], ys[i,1,j+1])]
                segments.append(segment)
                colors.append(colormap(colors_value[i, j]))
        lc = LineCollection(segments, colors=colors, linewidth=2, alpha=1/np.sqrt(ys.shape[0]))
        ax.add_collection(lc)
        
        if title is None:
            self.set_ax_style(ax, 'Aircraft Routes', legend=False)
        else:
            self.set_ax_style(ax, title, legend=False)
        

        if fig is not None:
            fig.tight_layout()
            # Save the plot to a file
            if filename is not None:
                fig.savefig(filename)

    def generate_confidence_region(self, means_x, means_y, sigma_x, sigma_y):
        ellipses = []
        for i in range(means_x.shape[0]):
            circle = Point(means_x[i], means_y[i]).buffer(1)
            ellipse = scale(circle, xfact=sigma_x[i], yfact=sigma_y[i], origin=(means_x[i], means_y[i]))
            ellipses.append(ellipse)
        union_ellipse = unary_union(ellipses)
        return union_ellipse

    def plot_multiple_routes_shaded(self, state_vectors_files, compressor, filename = None, ax = None, title = None):
        """
        Plots the routes of aircrafts with shaded regions representing deviations.
        
        Args:
        - state_vectors_files: a list of file paths containing state vectors for each aircraft.
        - compressor: a compressor object used to decode the state vectors.
        - filename (optional): the name of the file to save the plot to.
        - ax (optional): the axes to plot the data on.
        - title (optional): the title of the plot.
        """
        # Get the colormap to use for the plot
        colormap = getattr(plt.cm, self.options['cmap'])
        if ax is None:
            # Define the figure and axes for the plot
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                                figsize=[self.options['fig-size'][0], self.options['fig-size'][0]])
        else:
            fig = None
        # Add features to the plot
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linestyle=':')

        ys = np.zeros((len(state_vectors_files), 2, self.options['point-precision']))
        # Loop through each state vectors file and plot the route
        for n, file in enumerate(state_vectors_files):
            # Decode the state vectors from the file and interpolate to get a fixed number of points
            df = compressor.decode_to_dataframe_from_file(file)
            columns = ['lat', 'lon']
            df_interp = {col:[] for col in columns}
            df_interp = {'time':np.linspace(df['time'].iloc[0], df['time'].iloc[-1], num=self.options['point-precision'])}
            for col in columns:
                df_interp[col] = np.interp(df_interp['time'], df['time'], df[col])

            ys[n, 0, :] = df_interp['lon']
            ys[n, 1, :] = df_interp['lat']

        mask = ~np.any(np.isnan(ys), axis=(1, 2))
        ys = ys[mask]

        if self.options['expectation-measure'] == 'mean':
            expectation_name = 'Mean Path'
            expectation = np.mean(ys, axis=0)
        elif self.options['expectation-measure'] == 'median':
            expectation_name = 'Median Path'
            expectation = np.median(ys, axis=0)
        elif self.options['expectation-measure'] == 'average':
            expectation_name = 'Average Path'
            expectation = np.average(ys, axis=0)
        else:
            raise ValueError(f"Expectation Measure not recognized: {self.options['expectation-measure']}")

        sigmas = sorted(self.options['deviation-values'])[::-1]
        exp_lon, exp_lat = expectation
        if self.options['deviation-measure'] == 'std':
            sigma_names = [rf"${sigma:.1f}\sigma$ interval" for sigma in sigmas]
            sigma_lon = [sig*np.std(ys[:,0,:], axis = 0) for sig in sigmas]
            sigma_lat = [sig*np.std(ys[:,1,:], axis = 0) for sig in sigmas]
        elif self.options['deviation-measure'] == 'pct':
            sigma_names = [rf"${sigma:.1f}\%$ interval" for sigma in sigmas]
            sigma_lon = [(np.percentile(ys[:,0,:], 50 + sig/2, axis = 0) - np.percentile(ys[:,0,:], 50 - sig/2, axis = 0))/2 for sig in sigmas]
            sigma_lat = [(np.percentile(ys[:,1,:], 50 + sig/2, axis = 0) - np.percentile(ys[:,1,:], 50 - sig/2, axis = 0))/2 for sig in sigmas]
        else:
            raise ValueError(f"Deviation Measure not recognized: {self.options['deviation-measure']}")
        
        regions = [self.generate_confidence_region(exp_lon, exp_lat, sigma_lon[i], sigma_lat[i]) for i in range(len(sigmas))]
        
        # Plot the interpolated points
        for i, _ in enumerate(sigmas):
            color = np.array(colormap(float(i/len(sigmas))))
            x, y = regions[i].exterior.xy
            ax.fill(x, y, color = color,  alpha=0.5, label = sigma_names[i])
        ax.plot(expectation[0], expectation[1], color = colormap(1.), linewidth = 2, label = expectation_name)

        if title is None:
            self.set_ax_style(ax, 'Aircraft Route Distribution')
        else:
            self.set_ax_style(ax, title)
        
        if fig is not None:
            fig.tight_layout()
            # Save the plot to a file
            if filename is not None:
                fig.savefig(filename)
