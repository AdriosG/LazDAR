import sys
import os
import numpy as np
import laspy

from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg


def rasterize_dsm(x, y, z, cellsize):
    """
    Rasterize a MNS by height max per cell.
    x, y, z : 1D arrays
    cellsize : resolution (m)
    Return (dsm (ny,nx), xmin, ymin, cellsize)
    """
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    nx = int(np.ceil((xmax - xmin) / cellsize)) + 1
    ny = int(np.ceil((ymax - ymin) / cellsize)) + 1

    ix = np.floor((x - xmin) / cellsize).astype(np.int64)
    iy = np.floor((y - ymin) / cellsize).astype(np.int64)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(z)
    ix = ix[valid]
    iy = iy[valid]
    z = z[valid]

    dsm = np.full((ny, nx), -np.inf, dtype=np.float32)
    np.maximum.at(dsm, (iy, ix), z)
    dsm[dsm == -np.inf] = np.nan
    return dsm, xmin, ymin, cellsize



def hillshade(dem, cellsize, azimuth=315.0, altitude=45.0):
    """
    Hillshade :
      - azimuth in deg from N (0°)
      - altitude = Elevation of the virtual sun
      - returns an Array of 1..0 with NaN in empty cells
    """
    dem = np.asarray(dem, dtype=float)
    mask = np.isfinite(dem)
    if not np.any(mask):
        return np.full_like(dem, np.nan, dtype=np.float32)

    fill = np.nanmedian(dem[mask])
    dem_filled = np.where(mask, dem, fill)

    dy, dx = np.gradient(dem_filled, cellsize, cellsize)

    slope  = np.pi/2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)

    az  = np.deg2rad(float(azimuth))
    alt = np.deg2rad(float(altitude))

    hs = (np.sin(alt) * np.sin(slope) +
          np.cos(alt) * np.cos(slope) * np.cos(az - aspect))

    out = np.full_like(dem_filled, np.nan, dtype=np.float32)
    valid = mask & np.isfinite(hs)
    if np.any(valid):
        v = hs[valid]
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            out[valid] = (v - vmin) / (vmax - vmin)
        else:
            out[valid] = 0.5
    return out


class LidarViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lidar MNS Viewer (LAS/LAZ) – Zoom, Pan, Light")
        self.resize(1100, 750)

        self.dsm = None
        self.hill = None
        self.xmin = None
        self.ymin = None
        self.cellsize = 1.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        ctrl = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl, 0)

        self.btn_open = QtWidgets.QPushButton("Open LAS/LAZ…")
        self.btn_open.clicked.connect(self.open_file)
        ctrl.addWidget(self.btn_open)

        cell_layout = QtWidgets.QHBoxLayout()
        cell_layout.addWidget(QtWidgets.QLabel("Step (m) :"))
        self.spin_cell = QtWidgets.QDoubleSpinBox()
        self.spin_cell.setRange(0.1, 100.0)
        self.spin_cell.setSingleStep(0.1)
        self.spin_cell.setValue(1.0)
        self.spin_cell.valueChanged.connect(self.recompute_dsm)
        cell_layout.addWidget(self.spin_cell)
        ctrl.addLayout(cell_layout)

        az_layout = QtWidgets.QHBoxLayout()
        az_layout.addWidget(QtWidgets.QLabel("Azimut (°) :"))
        self.slider_az = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_az.setRange(0, 360)
        self.slider_az.setValue(315)
        self.slider_az.valueChanged.connect(self.update_hillshade)
        self.lbl_az = QtWidgets.QLabel("315")
        az_layout.addWidget(self.slider_az)
        az_layout.addWidget(self.lbl_az)
        ctrl.addLayout(az_layout)

        alt_layout = QtWidgets.QHBoxLayout()
        alt_layout.addWidget(QtWidgets.QLabel("Sun Elevation (°) :"))
        self.slider_alt = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_alt.setRange(5, 90)
        self.slider_alt.setValue(45)
        self.slider_alt.valueChanged.connect(self.update_hillshade)
        self.lbl_alt = QtWidgets.QLabel("45")
        alt_layout.addWidget(self.slider_alt)
        alt_layout.addWidget(self.lbl_alt)
        ctrl.addLayout(alt_layout)

        self.radio_dsm = QtWidgets.QRadioButton("View MNS (Colored)")
        self.radio_hs = QtWidgets.QRadioButton("View hillshade (Black and White)")
        self.radio_hs.setChecked(True)
        self.radio_dsm.toggled.connect(self.refresh_view)
        ctrl.addWidget(self.radio_hs)
        ctrl.addWidget(self.radio_dsm)

        self.chk_auto = QtWidgets.QCheckBox("Auto-contrast")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self.refresh_view)
        ctrl.addWidget(self.chk_auto)

        ctrl.addStretch(1)

        self.view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.view, 1)
        self.plot = self.view.addPlot()
        self.plot.invertY(True)
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.plot.showGrid(x=True, y=True, alpha=0.1)
        self.plot.setLabel('bottom', 'X (m)')
        self.plot.setLabel('left', 'Y (m)')

        try:
            self.cmap_dsm = pg.colormap.get('CET-L19')
        except Exception:
            self.cmap_dsm = pg.colormap.get('terrain')
        self.cmap_hs = pg.colormap.get('CET-L1')

    # ---------- Actions ----------
    def open_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open LAS/LAZ file", "", "Lidar (*.las *.laz)"
        )
        if not fn:
            return
        self.load_lidar(fn)

    def load_lidar(self, filepath):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            las = laspy.read(filepath)
            coords = np.asarray(las.xyz)

            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2].astype(np.float32, copy=False)
            

            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[good], y[good], z[good]

            self.cellsize = float(self.spin_cell.value())
            self.dsm, self.xmin, self.ymin, cs = rasterize_dsm(x, y, z, self.cellsize)
            self.update_hillshade()
            self.zoom_full()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Loading Failded: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def recompute_dsm(self):
        if self.dsm is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.cellsize = float(self.spin_cell.value())
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.information(
                self, "Info",
                "To recalculate MNS with a different step value, please re-open the file."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def update_hillshade(self):
        if self.dsm is None:
            return
        az = int(self.slider_az.value())
        alt = int(self.slider_alt.value())
        self.lbl_az.setText(str(az))
        self.lbl_alt.setText(str(alt))

        mask = np.isfinite(self.dsm)
        if not np.any(mask):
            return
        hs = np.empty_like(self.dsm, dtype=np.float32)
        hs[:] = np.nan
        try:
            hs[mask] = hillshade(self.dsm[mask].reshape(-1, 1).reshape(self.dsm.shape),
                                 self.cellsize, azimuth=az, altitude=alt)[mask]
        except Exception:
            hs = hillshade(self.dsm, self.cellsize, azimuth=az, altitude=alt)
        self.hill = hs
        self.refresh_view()

    def refresh_view(self):
        if self.dsm is None:
            return
        show_hs = self.radio_hs.isChecked()

        if show_hs and self.hill is not None:
            img = self.hill
            cmap = self.cmap_hs
        else:
            img = self.dsm
            cmap = self.cmap_dsm

        h, w = img.shape
        x0 = self.xmin
        y0 = self.ymin
        cs = self.cellsize

        self.img_item.setImage(img.T, autoLevels=self.chk_auto.isChecked())
        self.img_item.setLookupTable(cmap.getLookupTable(nPts=512))
        self.img_item.resetTransform()
        self.img_item.setPos(x0, y0)
        self.img_item.setScale(cs)

    def zoom_full(self):

        self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = LidarViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
