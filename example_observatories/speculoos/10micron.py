"""
10Micron GM-HPS Mount – Astra Observatory Subclass
===================================================

Builds a pointing model via the mount's native :newalig/:newalpt/:endalig
pipeline over a dedicated TCP connection (as MountWizzard4 does).

Conventions:
  - MRA/MDEC/pier are read directly from the mount over TCP (:Ginfo#) in JNow,
    so there is no ASCOM caching and no J2000 round trip. While the mount
    tracks, they do not change, so reading them after the plate solve is fine.
  - SIDTIME is the mount's sidereal time (:GS#) stepped back to mid-exposure.
    The read happens after the plate solve, and every second of delay would
    otherwise add 15" of hour-angle error to the point.
  - PRA/PDEC are the plate-solved frame centre (ICRS) transformed to TETE
    (true equator, true equinox, i.e. the mount's JNow) at mid-exposure,
    with no refraction.
  - :newalpt fields use the formats of the 10Micron command protocol
    (firmware 3.2.5): RA and SIDTIME as HH:MM:SS.S, Dec as sDD:MM:SS.
  - Point capture is gated by a model-building flag, not by sync=True, so
    ordinary Astra centring/sync calls pass through untouched.
"""

from __future__ import annotations

import math
import socket
from datetime import datetime
from pathlib import Path

import astropy.units as u
from astropy.coordinates import TETE, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from astra.observatory import Observatory
from astra.paired_devices import PairedDevices

MOUNT_IP = "192.168.1.11"  # used when the Telescope config has no "mount_ip"
MOUNT_PORT = 3490
MAX_CLOCK_OFFSET_MS = 100.0
SIDEREAL_PER_SOLAR = 1.00273790935  # sidereal seconds per SI second


class AlignmentPoint:
    __slots__ = (
        "mount_dec",
        "mount_ra",
        "pier_side",
        "sidereal_time",
        "solved_dec",
        "solved_ra",
    )

    def __init__(
        self, mount_ra, mount_dec, pier_side, solved_ra, solved_dec, sidereal_time
    ):
        self.mount_ra = mount_ra
        self.mount_dec = mount_dec
        self.pier_side = pier_side
        self.solved_ra = solved_ra
        self.solved_dec = solved_dec
        self.sidereal_time = sidereal_time


class MountTCPError(RuntimeError):
    """The TCP link to the mount failed. The connection is closed."""


class MountTCP:
    """Direct TCP connection to a 10Micron mount."""

    def __init__(
        self, host: str, port: int = MOUNT_PORT, timeout: float = 10.0, logger=None
    ):
        self.host, self.port, self.timeout, self.logger = host, port, timeout, logger
        self.sock: socket.socket | None = None

    def connect(self) -> bool:
        try:
            self.sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
            return True
        except OSError as e:
            if self.logger:
                self.logger.error(f"10Micron TCP: connect failed: {e}")
            self.sock = None
            return False

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _fail(self, message: str) -> MountTCPError:
        # A missing or late reply would be read as the answer to the next command,
        # so the connection cannot be used any more.
        self.close()
        if self.logger:
            self.logger.error(f"10Micron TCP: {message}")
        return MountTCPError(f"10Micron TCP: {message}")

    def send_blind(self, cmd: str) -> None:
        if self.sock is None:
            raise MountTCPError("10Micron TCP: not connected")
        try:
            self.sock.sendall(f":{cmd}#".encode("ascii"))
        except OSError as e:
            raise self._fail(f"send of :{cmd}# failed: {e}") from e

    def send_cmd(self, cmd: str) -> str:
        """Send a command whose reply ends with '#'; return the reply without it."""
        self.send_blind(cmd)
        buf = b""
        try:
            while not buf.endswith(b"#"):
                chunk = self.sock.recv(1024)
                if not chunk:
                    raise self._fail(f"the mount closed the connection during :{cmd}#")
                buf += chunk
        except TimeoutError as e:
            raise self._fail(f"no reply to :{cmd}# within {self.timeout:.0f} s") from e
        except OSError as e:
            raise self._fail(f"receive after :{cmd}# failed: {e}") from e
        resp = buf.decode("ascii", errors="replace")[:-1]
        if self.logger:
            self.logger.debug(f"10Micron TCP :{cmd}# -> '{resp}'")
        return resp

    def send_char(self, cmd: str) -> str:
        """Send a command whose reply is exactly one character (for example :GREF#)."""
        self.send_blind(cmd)
        try:
            data = self.sock.recv(1)
        except TimeoutError as e:
            raise self._fail(f"no reply to :{cmd}# within {self.timeout:.0f} s") from e
        except OSError as e:
            raise self._fail(f"receive after :{cmd}# failed: {e}") from e
        if not data:
            raise self._fail(f"the mount closed the connection during :{cmd}#")
        return data.decode("ascii", errors="replace")


class TenMicron(Observatory):
    OBSERVATORY_ALIASES: list[str] = [
        "10micron",
        "gm1000",
        "gm2000",
        "gm3000",
        "gm4000",
    ]

    _building_model = False

    # ---- formatting / parsing ------------------------------------------

    @staticmethod
    def _sexa(s: str) -> float:
        """'sDD:MM:SS.S' / 'sDD*MM:SS.S' / 'HH:MM:SS.SS' -> decimal."""
        s = s.replace("*", ":")
        sign = -1.0 if s.startswith("-") else 1.0
        d, m, sec = (float(x) for x in s.lstrip("+-").split(":"))
        return sign * (d + m / 60 + sec / 3600)

    @staticmethod
    def _split(x: float, dp: int) -> tuple[int, int, float]:
        """Round first, then split, so seconds never come out as 60."""
        tot = round(abs(x) * 3600, dp)
        d, rem = divmod(tot, 3600)
        m, s = divmod(rem, 60)
        return int(d), int(m), s

    def _fmt_ra(self, hours: float) -> str:
        """HH:MM:SS.S, as :newalpt expects for MRA, PRA and SIDTIME."""
        h, m, s = self._split(hours % 24.0, 1)
        h %= 24
        return f"{h:02d}:{m:02d}:{s:04.1f}"

    def _fmt_dec(self, deg: float) -> str:
        """sDD:MM:SS (whole arcseconds), as :newalpt expects for MDEC and PDEC."""
        d, m, s = self._split(deg, 0)
        sign = "-" if deg < 0 and (d or m or s) else "+"
        return f"{sign}{d:02d}:{m:02d}:{int(s):02d}"

    # ---- mount state over TCP -----------------------------------------

    def _init_mount_session(self, tcp: MountTCP) -> None:
        tcp.send_blind("U2")  # ultra precision for this session only
        lat = self._sexa(tcp.send_cmd("Gt"))
        lon = -self._sexa(
            tcp.send_cmd("Gg")
        )  # the mount counts east longitudes as negative
        elev = float(tcp.send_cmd("Gev"))
        self._location = EarthLocation(
            lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m
        )

        offset_ms = (float(tcp.send_cmd("GJD1")) - Time.now().jd) * 86400e3
        msg = f"10Micron: mount clock - PC clock = {offset_ms:+.0f} ms"
        if abs(offset_ms) > MAX_CLOCK_OFFSET_MS:
            self.logger.warning(
                msg + " — sync the mount clock before building a model."
            )
        else:
            self.logger.info(msg)

        # The solved coordinates have no refraction. The mount adds refraction with its own
        # temperature and pressure, so wrong values there end up in the model.
        refraction = tcp.send_char("GREF")
        temperature = tcp.send_cmd("GRTMP")
        pressure = tcp.send_cmd("GRPRS")
        msg = f"10Micron: refraction {'on' if refraction == '1' else 'OFF'}, {temperature} °C, {pressure} hPa"
        if refraction != "1":
            self.logger.warning(msg + " — turn refraction on before building a model.")
        else:
            self.logger.info(msg + " (check these match the current conditions)")

    def _read_mount(self) -> dict:
        """JNow RA/Dec, pointing state and LST (hours), with the PC time of the LST read."""
        f = self._tcp.send_cmd("Ginfo").split(",")
        if len(f) < 8:
            raise MountTCPError(
                f"10Micron TCP: unexpected :Ginfo# reply {','.join(f)!r}"
            )
        t = Time.now()
        lst = self._sexa(self._tcp.send_cmd("GS"))  # HH:MM:SS.SS in ultra precision
        return {
            "ra": float(f[0]),
            "dec": float(f[1]),
            "side": f[2],
            "lst": lst,
            "time": t,
        }

    # ---- solved centre -> JNow -----------------------------------------

    def _solved_apparent(self, wcs: WCS, filepath) -> tuple[float, float, Time]:
        """Solved frame centre in TETE (JNow) and the mid-exposure time."""
        if filepath is None:
            raise ValueError("no image file: the exposure time is unknown")
        header = fits.getheader(filepath)
        # Use the image size from the FITS header. Do NOT use wcs.pixel_shape: the WCS
        # comes from astropy's fit_wcs_from_points (via twirl), which sets pixel_shape to
        # the largest x/y of the matched stars, not to the image size. That "centre" moves
        # from frame to frame and put arcminutes of random error into every point.
        width, height = header["NAXIS1"], header["NAXIS2"]
        centre = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)

        # Astra writes DATE-OBS as the UTC start of the exposure.
        t_mid = (
            Time(header["DATE-OBS"], scale="utc") + float(header["EXPTIME"]) / 2 * u.s
        )

        app = SkyCoord(centre).icrs.transform_to(
            TETE(obstime=t_mid, location=self._location)
        )
        return app.ra.hour, app.dec.deg, t_mid

    # ---- main sequence -------------------------------------------------

    def _get_mount_ip(self, paired_devices: PairedDevices) -> str:
        return paired_devices.get_device_config("Telescope").get("mount_ip", MOUNT_IP)

    def pointing_model_sequence(self, action, paired_devices: PairedDevices) -> None:
        self._alignment_points: list[AlignmentPoint] = []
        host = self._get_mount_ip(paired_devices)
        tcp = MountTCP(host, logger=self.logger)
        if not tcp.connect():
            raise RuntimeError(
                f"10Micron: cannot connect to mount at {host}:{MOUNT_PORT}."
            )
        self._tcp = tcp

        try:
            self._init_mount_session(tcp)

            if tcp.send_cmd("newalig") != "V":
                raise RuntimeError("10Micron: :newalig# failed.")

            self._building_model = True
            try:
                super().pointing_model_sequence(action, paired_devices)
            finally:
                self._building_model = False

            n = len(self._alignment_points)
            if n < 3:
                self.logger.warning(
                    f"10Micron: only {n} points — existing model retained."
                )
                return

            accepted = 0
            for i, p in enumerate(self._alignment_points, 1):
                resp = tcp.send_cmd(
                    f"newalpt{p.mount_ra},{p.mount_dec},{p.pier_side},"
                    f"{p.solved_ra},{p.solved_dec},{p.sidereal_time}"
                )
                if resp == "E":
                    self.logger.warning(f"  point {i}/{n} rejected")
                else:
                    accepted += 1
            self.logger.info(f"10Micron: {accepted}/{n} points accepted.")
            if accepted < 3:
                self.logger.warning(
                    "10Micron: fewer than 3 points accepted — existing model retained."
                )
                return

            if tcp.send_cmd("endalig") != "V":
                self.logger.warning(
                    "10Micron: :endalig# failed — existing model retained."
                )
                return

            name = f"A_{datetime.now():%y%m%d_%H%M}"
            tcp.send_cmd(f"modeldel0{name}")
            saved = tcp.send_cmd(f"modelsv0{name}") == "1"
            self.logger.info(
                f"10Micron: model active ({tcp.send_cmd('getalst')} stars), "
                f"{'saved as ' + name if saved else 'SAVE FAILED'}; "
                f"getain: {tcp.send_cmd('getain')}"
            )
        finally:
            tcp.close()
            self._tcp = None

    # ---- capture data pairs ---------------------------------------------

    def pointing_correction(
        self,
        action,
        filepath: str | Path | None,
        paired_devices: PairedDevices,
        dark_frame: str | Path | None = None,
        sync: bool = False,
        slew: bool = True,
    ) -> tuple[bool, WCS | None]:
        if not self._building_model:
            return super().pointing_correction(
                action,
                filepath,
                paired_devices,
                dark_frame=dark_frame,
                sync=sync,
                slew=slew,
            )

        # Astra calls this after the exposure. With sync=False and slew=False the base
        # class only plate-solves; it does not move the mount.
        before = self._read_mount()
        ok, wcs = super().pointing_correction(
            action,
            filepath,
            paired_devices,
            dark_frame=dark_frame,
            sync=False,
            slew=False,
        )
        if wcs is None:
            return ok, wcs

        # While tracking, RA/Dec must not change between the two reads. A difference
        # means the mount moved (for example tracking stopped at a limit) during the solve.
        mount = self._read_mount()
        d_ra_s = (
            ((mount["ra"] - before["ra"] + 12) % 24 - 12)
            * 54000
            * math.cos(math.radians(mount["dec"]))
        )
        moved = math.hypot(d_ra_s, (mount["dec"] - before["dec"]) * 3600)
        if moved > 2.0 or mount["side"] != before["side"]:
            self.logger.warning(
                f'10Micron: mount moved {moved:.1f}" during the solve - point skipped'
            )
            return ok, wcs

        try:
            s_ra, s_dec, t_mid = self._solved_apparent(wcs, filepath)
        except (KeyError, ValueError) as e:
            self.logger.warning(
                f"10Micron: cannot get the mid-exposure time ({e}) - point skipped"
            )
            return ok, wcs

        # Step the mount's sidereal time back from the read to mid-exposure. Both times
        # come from the PC clock, so the mount clock offset does not enter.
        dt = (mount["time"] - t_mid).to_value(u.s)
        lst_mid = (mount["lst"] - dt * SIDEREAL_PER_SOLAR / 3600) % 24

        p = AlignmentPoint(
            mount_ra=self._fmt_ra(mount["ra"]),
            mount_dec=self._fmt_dec(mount["dec"]),
            pier_side=mount["side"],
            solved_ra=self._fmt_ra(s_ra),
            solved_dec=self._fmt_dec(s_dec),
            sidereal_time=self._fmt_ra(lst_mid),
        )
        self._alignment_points.append(p)

        d_ra = (s_ra - mount["ra"] + 12) % 24 - 12
        self.logger.info(
            f"10Micron: point #{len(self._alignment_points)} pier={p.pier_side} "
            f'dRA={d_ra * 54000 * math.cos(math.radians(mount["dec"])):+.1f}" '
            f'dDec={(s_dec - mount["dec"]) * 3600:+.1f}" '
            f"(solve took {dt:.1f} s after mid-exposure)"
        )
        return ok, wcs
