# Python 3 compatibility
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import nibabel as nib
import stat159lambda.utils.scene_slicer as ssm
import stat159lambda.utils.data_path as dp
from stat159lambda.config import REPO_HOME_PATH, NUM_OFFSET_VOLUMES


class VoxelExtractor:
    def __init__(self, subject, interest_col_str, data=None):
        """
        VoxelExtractor generates the t-statistic for each voxel based on a
        given interest column. For example, to see the voxel that can best
        predict interior or exterior, you would use the following code.

        ve = VoxelExtractor(1, 'int-ext')
        a = ve.t_stat()[0]
        ve.plot_single_voxel(0)

        interest_col_str takes in either 'day-night' or 'int-ext'.
        data can also be overrided if needed, else the smoothed 2d version
        will be used.
        """
        self.subject = subject
        if interest_col_str == "int-ext":
            self.interest_col_ind = 0
        elif interest_col_str == "day-night":
            self.interest_col_ind = 1
        else:
            raise ValueError("Incorrect interest column name: please use either 'int-ext' or 'day-night'")
        self.interest_col_str = interest_col_str
        if data is None:
            data_path = dp.get_smoothed_2d_path(self.subject, 4)
            data = np.load(data_path)
            data = data[:, NUM_OFFSET_VOLUMES:]
        self.data = data
        self.design = None
        self.B = None
        self.t_values = None

    def get_design_matrix(self):
        """
        Returns
        -------
        Design matrix with 3 columns, including the column of interest,
        the linear drift column, and the column of ones
        """
        if self.design is None:
            scene_path = dp.get_scene_csv()
            ss = ssm.SceneSlicer(scene_path)
            interest_col = ss.get_scene_slices()[self.interest_col_ind]
            n_trs = self.data.shape[-1]
            design = np.ones((n_trs, 3))
            design[:, 1] = np.linspace(-1, 1, n_trs)
            design[:, 2] = interest_col[NUM_OFFSET_VOLUMES:NUM_OFFSET_VOLUMES+n_trs]
            self.design = design
        return self.design

    def plot_design_matrix(self):
        """
        Saves the design matrix as an image into the figures folder.
        Returns
        -------
        None
        """
        if self.design is None:
            self.get_design_matrix()
        design_fig = plt.gcf()
        plt.imshow(self.design,
                   aspect=0.1,
                   cmap='gray',
                   interpolation='nearest')
        plt.xticks([])
        design_fig_path = '{0}/figures/design_fig_{1}.png'.format(
            REPO_HOME_PATH, self.interest_col_str)
        design_fig.savefig(design_fig_path, dpi=100)
        plt.clf()

    def get_betas_Y(self):
        """
        Returns
        -------
        B: 2D array, p x B, the number of voxels
        """
        if self.design is None:
            self.get_design_matrix()
        if not self.B:
            Y = self.data
            self.B = npl.pinv(self.design).dot(Y.T)
        return self.B

    def t_stat(self):
        """ betas, t statistic and significance test given data,
        design matix, contrast
        This is OLS estimation; we assume the errors to have independent
        and identical normal distributions around zero for each $i$ in
        $\e_i$ (i.i.d).
        """
        if self.design is None:
            self.get_design_matrix()
        if self.t_values is None:
            y = self.data.T
            X = self.design
            c = [0, 0, 1]
            c = np.atleast_2d(c).T
            beta = npl.pinv(X).dot(y)
            fitted = X.dot(beta)
            errors = y - fitted
            RSS = (errors**2).sum(axis=0)
            df = X.shape[0] - npl.matrix_rank(X)
            MRSS = RSS / df
            SE = np.sqrt(MRSS * c.T.dot(npl.pinv(X.T.dot(X)).dot(c)))
            try:
                SE[SE == 0] = np.amin(SE[SE != 0])
            except ValueError:
                pass
            t = c.T.dot(beta) / SE
            self.t_values = abs(t[0])
        self.t_indices = np.array(self.t_values).argsort(
        )[::-1][:self.t_values.size]
        return self.t_indices

    def plot_single_voxel(self, voxel_index):
        """
        Plots a single voxel timecourse and the figure is saved under the
        figures folder.
        Returns
        -------
        None
        """
        voxel_img = plt.gcf()
        plt.plot(self.data[voxel_index, :])
        voxel_img_path = '{0}/figures/voxel_{1}.png'.format(REPO_HOME_PATH,
                                                            voxel_index)
        voxel_img.savefig(voxel_img_path, dpi=100)
        plt.clf()
