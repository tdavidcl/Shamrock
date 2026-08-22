"""
Shamrock plot utility functions.
"""

import glob
import io

import shamrock.sys

__all__ = []

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    # print("Warning: matplotlib is not installed, some Shamrock functions will not be available")

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    # print("Warning: PIL is not installed, some Shamrock functions will not be available")

try:
    import graphviz

    _HAS_GRAPHVIZ = True
except ImportError:
    _HAS_GRAPHVIZ = False
    # print("Warning: graphviz is not installed, some Shamrock functions will not be available")

if _HAS_MATPLOTLIB and _HAS_PIL:
    __all__.append("show_image_sequence")

    def show_image_sequence(
        glob_str, render_gif=True, dpi=200, interval=50, repeat_delay=10, fig=None
    ):
        """
        Create a matplotlib animation from a sequence of image files.

        Available only if matplotlib and PIL are installed.

        Parameters
        ----------
        glob_str : str
            Glob pattern matching image files.
        render_gif : bool, optional
            Whether to render the animation.
        dpi : int, optional
            Dots per inch for the figure.
        interval : int, optional
            Delay between frames in milliseconds.
        repeat_delay : int, optional
            Delay before repeating the animation.

        Raises
        ------
        FileNotFoundError : if no images are found for the glob pattern

        Returns
        -------
        matplotlib.animation.FuncAnimation or None
            Animation object on rank 0, otherwise None.
        """

        if not render_gif:
            return None

        if shamrock.sys.world_rank() != 0:
            return None

        files = sorted(glob.glob(glob_str))

        image_array = []
        for my_file in files:
            with Image.open(my_file) as image:
                image_array.append(image.copy())

        if not image_array:
            raise FileNotFoundError(f"No images found for glob pattern: {glob_str}")

        pixel_x, pixel_y = image_array[0].size

        if fig is None:
            fig = plt.figure(dpi=dpi)
        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
        plt.axis("off")

        im = plt.imshow(image_array[0], animated=True, aspect="auto")

        def update(i):
            im.set_array(image_array[i])
            return (im,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(image_array),
            interval=interval,
            blit=True,
            repeat_delay=repeat_delay,
        )

        return ani


if _HAS_MATPLOTLIB and _HAS_GRAPHVIZ:
    __all__.append("show_dot_graph")

    def show_dot_graph(dot_source, dpi=150):
        """
        Render a Graphviz DOT graph as a matplotlib figure.

        Useful to display the solver graphs produced by e.g.
        ``model.get_solver_dot_graph()`` or ``setup_node.get_dot()`` directly
        in the sphinx-gallery generated examples, since figures created this
        way are automatically picked up by sphinx-gallery's matplotlib
        scraper.

        Available only if matplotlib and the graphviz python package (plus
        the Graphviz ``dot`` executable) are installed.

        Parameters
        ----------
        dot_source : str
            Source of the graph in the DOT language.
        dpi : int, optional
            Dots per inch used to size the resulting figure.

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure on rank 0, otherwise None.
        """

        if shamrock.sys.world_rank() != 0:
            return None

        png_bytes = graphviz.Source(dot_source).pipe(format="png")

        with io.BytesIO(png_bytes) as buf:
            image = plt.imread(buf, format="png")

        height, width = image.shape[:2]

        fig = plt.figure(dpi=dpi, figsize=(width / dpi, height / dpi))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(image)
        ax.axis("off")

        return fig
