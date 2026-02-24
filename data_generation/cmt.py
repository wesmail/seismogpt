import numpy as np
from scipy.special import erf


class CMT:
    """
    Class to handle randomly generated moment tensors for augmenting
    synthetic seismic datasets.
    """

    def __init__(self, **kwargs):

        # Select the exponent for moment tensor
        # Either random, or approximate a value based on the user's Mw input
        user_Mw = kwargs.get("Mw", None)
        self.exp = self.select_exponent(user_Mw)
        # Get moment tensor elements, units are Nm
        self.rmt = self.random_moment_tensor()

        # Full moment tensor in Nm
        self.mt = np.array(
            [
                [self.rmt[0], self.rmt[3], self.rmt[4]],
                [self.rmt[3], self.rmt[1], self.rmt[5]],
                [self.rmt[4], self.rmt[5], self.rmt[2]],
            ]
        )
        # Seismic moment in Nm
        self.M0 = (1.0 / np.sqrt(2.0)) * np.linalg.norm(self.mt)
        # Moment magnitude
        self.Mw = (2.0 / 3.0) * np.log10(self.M0 * 1e7) - 10.7

        # If the user asked for a specific Mw, an exponent will have been approximated
        # but due to randomness this is unlikely to give the exact requsted Mw.
        # In this case, normalise the moment tensor to the requested Mw.
        if user_Mw is not None and abs(self.Mw - user_Mw) >= 1e-5:
            self.normalise_mt(user_Mw)

        # Elements of the moment tensor in Nm
        self.mrr = self.rmt[0]
        self.mtt = self.rmt[1]
        self.mpp = self.rmt[2]
        self.mrt = self.rmt[3]
        self.mrp = self.rmt[4]
        self.mtp = self.rmt[5]

        # Half shift and time duration, both in seconds
        self.ts = kwargs.get("ts", self.random_time_shift())
        self.hd = kwargs.get("hd", self.calc_half_duration())

        # Source location
        location = self.random_location()
        self.lat = kwargs.get("lat", location[0])  # Degrees N
        self.lon = kwargs.get("lon", location[1])  # Degrees E
        self.dep = kwargs.get("dep", self.random_depth())  # km
        self.location = np.array([self.lat, self.lon, self.dep])

    def write(self, filename="CMTSOLUTION", directory="./"):
        """
        Function to write text file in CMTSOLUTION format
        """

        # The contents of the file
        string = (
            " ABCD 0000 00 00 00 00 00.00  0.0000 "
            + "000.0000 0.0 0.0 0.0 EFGHIJKLMNOPQ\n"
            + f"event name:     000000000000X\n"
            + f"time shift:   {self.ts:>9.4f}\n"
            + f"half duration:{self.hd:>9.4f}\n"
            + f"latitude:     {self.lat:>9.4f}\n"
            + f"longitude:    {self.lon:>9.4f}\n"
            + f"depth:        {self.dep:>9.4f}\n"
            + f"Mrr:      {self.mrr * 1e7:>13.6E}\n"
            + f"Mtt:      {self.mtt * 1e7:>13.6E}\n"
            + f"Mpp:      {self.mpp * 1e7:>13.6E}\n"
            + f"Mrt:      {self.mrt * 1e7:>13.6E}\n"
            + f"Mrp:      {self.mrp * 1e7:>13.6E}\n"
            + f"Mtp:      {self.mtp * 1e7:>13.6E}"
        )

        # Write file
        f = open(directory + filename, "w")
        f.write(string)
        f.close()

    def random_moment_tensor(self, factor=None):
        """
        Function to generate random moment tensor components.

        Some key points:
            - mtt, mpp, mrt, mrp, mtp are treated as independent Gaussians,
              the parameters of which depend on the exponent.
            - The sum of mrr, mtt and mpp must equal zero for a source with no
              volume change, so these are generated and then demeaned together.

        Returns the reduced moment tensor:
            [mrr, mtt, mpp, mrt, mrp, mtp]
        """

        # Treat mtt, mpp, mrt, mrp and mtp as independent Gaussians
        x = np.linspace(-10.0, 10.0, 1001)
        params = [
            self.get_Gauss_params(m, self.exp)
            for m in ["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]
        ]
        pdfs = [self.Gaussian(x, *p) for p in params]
        ms = np.array([np.random.choice(x, p=pdf / np.sum(pdf)) for pdf in pdfs])

        # Demean mrr, mtt and mpp
        # This ensures that they sum to zero
        ms = np.append(ms[:3] - np.mean(ms[:3]), ms[3:])

        return np.array(ms) * 10.0**self.exp

    def random_location(self):
        """
        Function generates a random source latitude and longitude.

        Latitude and longitude are independent of the moment tensor.

        Latitude and longitude are treated as uniform. This isn't true for the
        real Earth, but the synthetics don't have 3D variations anyway.
        """

        # Random latitude and longitude
        latitude = np.random.uniform(-90, 90)
        longitude = np.random.uniform(-180, 180)

        return latitude, longitude

    def random_depth(self):
        """
        Function generates a random source depth.

        Depth is independent of the moment tensor.

        For geological reasons the depth distribution of earthquakes is complex
        and is not well fit by standard distributions. This is the distribution is from
        all events in the GCMT Catalogue from 2004 - 2020 in 100 equal width bins.
        """

        # Depth bins
        x = np.array(
            [
                0.0,
                7.0,
                14.0,
                21.0,
                28.0,
                35.0,
                42.0,
                49.0,
                56.0,
                63.0,
                70.0,
                77.0,
                84.0,
                91.0,
                98.0,
                105.0,
                112.0,
                119.0,
                126.0,
                133.0,
                140.0,
                147.0,
                154.0,
                161.0,
                168.0,
                175.0,
                182.0,
                189.0,
                196.0,
                203.0,
                210.0,
                217.0,
                224.0,
                231.0,
                238.0,
                245.0,
                252.0,
                259.0,
                266.0,
                273.0,
                280.0,
                287.0,
                294.0,
                301.0,
                308.0,
                315.0,
                322.0,
                329.0,
                336.0,
                343.0,
                350.0,
                357.0,
                364.0,
                371.0,
                378.0,
                385.0,
                392.0,
                399.0,
                406.0,
                413.0,
                420.0,
                427.0,
                434.0,
                441.0,
                448.0,
                455.0,
                462.0,
                469.0,
                476.0,
                483.0,
                490.0,
                497.0,
                504.0,
                511.0,
                518.0,
                525.0,
                532.0,
                539.0,
                546.0,
                553.0,
                560.0,
                567.0,
                574.0,
                581.0,
                588.0,
                595.0,
                602.0,
                609.0,
                616.0,
                623.0,
                630.0,
                637.0,
                644.0,
                651.0,
                658.0,
                665.0,
                672.0,
                679.0,
                686.0,
                693.0,
            ]
        )

        # Bin occurence
        y = np.array(
            [
                7.6300e02,
                1.5951e04,
                2.2520e03,
                1.9620e03,
                2.0620e03,
                3.0920e03,
                1.3180e03,
                9.4700e02,
                7.1900e02,
                5.3500e02,
                3.8800e02,
                3.1600e02,
                3.0200e02,
                2.8800e02,
                3.3600e02,
                3.3900e02,
                3.0600e02,
                2.7300e02,
                2.2700e02,
                1.8300e02,
                1.7700e02,
                1.5400e02,
                1.4300e02,
                1.2200e02,
                9.8000e01,
                8.3000e01,
                8.5000e01,
                9.3000e01,
                8.0000e01,
                7.4000e01,
                7.8000e01,
                6.1000e01,
                5.3000e01,
                3.7000e01,
                2.8000e01,
                2.6000e01,
                3.0000e01,
                2.1000e01,
                2.0000e01,
                1.1000e01,
                1.7000e01,
                1.7000e01,
                1.0000e01,
                9.0000e00,
                1.3000e01,
                1.2000e01,
                8.0000e00,
                6.0000e00,
                1.0000e01,
                7.0000e00,
                1.7000e01,
                1.1000e01,
                2.2000e01,
                1.1000e01,
                1.5000e01,
                2.4000e01,
                1.5000e01,
                2.4000e01,
                1.8000e01,
                1.0000e01,
                1.2000e01,
                8.0000e00,
                1.4000e01,
                1.0000e01,
                6.0000e00,
                1.0000e01,
                1.2000e01,
                1.1000e01,
                1.5000e01,
                1.5000e01,
                2.1000e01,
                2.1000e01,
                3.2000e01,
                2.2000e01,
                2.3000e01,
                3.8000e01,
                4.1000e01,
                3.8000e01,
                4.6000e01,
                4.0000e01,
                3.9000e01,
                4.5000e01,
                4.7000e01,
                5.1000e01,
                2.8000e01,
                4.5000e01,
                4.6000e01,
                2.8000e01,
                2.8000e01,
                1.9000e01,
                1.3000e01,
                1.2000e01,
                6.0000e00,
                7.0000e00,
                3.0000e00,
                4.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                0.0000e00,
            ]
        )

        # Take random value from the depth distribution
        # Add some scatter that is between the sampling intervals
        pdf = y / np.sum(y)
        depth = np.random.choice(x, p=pdf) + np.random.uniform(0, x[1] - x[0])

        return depth

    def random_time_shift(self):
        """
        Function generates a random time shift.

        Time shifts are well represented by a Gaussian and are essentially independent
        of the moment tensor.
        """

        min_ts = -5.0
        max_ts = 15.0
        params = np.array([2.7852479526842586, 2.9826431791841683])
        x = np.arange(min_ts, max_ts, 0.1)
        pdf = self.Gaussian(x, *params)
        return np.random.choice(x, p=pdf / np.sum(pdf))

    def calc_half_duration(self):
        """
        Function calculates the half duration of the source.
        Half duration is fixed prior to the GCMT inversion.

        For more details, see:
          Ekström, G., M. Nettles, and A. M. Dziewonski, The global CMT project
          2004-2010: Centroid-moment tensors for 13,017 earthquakes,
          Phys. Earth Planet. Inter., 200-201, 1-9, 2012.
          doi:10.1016/j.pepi.2012.04.002
        """

        hd = 1.05e-8 * ((self.M0 * 1e7) ** (1.0 / 3.0))

        return hd

    def select_exponent(self, Mw):
        """
        Function selects an exponent based on whether there is a user input Mw or not.
        Either generates a random one, or estimates and exponent based on the Mw.
        """

        if Mw is None:
            return self.random_exponent()
        else:
            return np.log10(np.sqrt(2.0) / 3.0 * 10.0 ** (1.5 * (Mw + 10.7))) - 7.0

    def random_exponent(self):
        """
        Function generates an exponent for the moment tensor components.
        The exponent is drawn from a skewed Gaussian fitted to the GCMT catalogue.
        """

        min_exp = 14.0
        max_exp = 25.0
        params = np.array([4.039460314907856, 15.646139738204411, 0.9880063620183555])
        x = np.arange(min_exp, max_exp, 0.01)
        pdf = self.skewed_Gaussian(x, *params)
        return np.random.choice(x, p=pdf / np.sum(pdf))

    def normalise_mt(self, Mw):
        """
        If the user chose an Mw then an exponent will have been estimated, but due
        to randomness the resulting moment tensor is very unlikely to be the correct
        magnitude.

        This function normalised the generated moment tensor to ensure that it is the
        magnitude that the user requested.
        """

        # Target norm based on requested Mw
        M = (np.sqrt(2.0) * 10.0 ** (1.5 * (Mw + 10.7))) * 1e-7

        # Normalise the moment tensor and assign new attributes
        self.mt = self.mt * (M / np.linalg.norm(self.mt))
        self.M0 = (1.0 / np.sqrt(2.0)) * np.linalg.norm(self.mt)
        self.Mw = (2.0 / 3.0) * np.log10(self.M0 * 1e7) - 10.7

    def get_Gauss_params(self, component, exp):
        """
        Function gives the parameters of a Gaussian that fits each moment tensor
        component for a given exponent.

        These are calculated by fitting the distribution of moment tensor component
        coefficient in the GCMT catalogue 2004 - 2020.
        """

        coefficients = {
            "mrr": {
                15.0: (-0.43036055776892435, 2.047965493077154),
                16.0: (0.09265761392460423, 2.6083540015857767),
                17.0: (0.45267795301643315, 1.963168220214603),
                18.0: (0.5465493439817456, 1.9382676729883297),
                19.0: (0.388461126005362, 2.0983292720027293),
                20.0: (0.44690322580645164, 2.0977319757890087),
            },
            "mtt": {
                15.0: (0.10004581673306773, 1.495010619654893),
                16.0: (-0.056428408991515785, 2.081198026109688),
                17.0: (-0.15642770486379176, 1.6882339242827726),
                18.0: (-0.2264164289788933, 1.6974702680458096),
                19.0: (-0.31248793565683647, 1.5174109486621832),
                20.0: (-0.33756989247311825, 1.4329387308623367),
            },
            "mpp": {
                15.0: (0.3301812749003984, 1.8938101114039247),
                16.0: (-0.0362171346103385, 2.3814216247670243),
                17.0: (-0.2961677511856182, 1.8238043879636214),
                18.0: (-0.3202036508841985, 1.8893751434193298),
                19.0: (-0.07594638069705097, 1.8832770202300506),
                20.0: (-0.10950537634408607, 1.8799522278513765),
            },
            "mrt": {
                15.0: (0.0758316733067729, 0.8350498175329074),
                16.0: (0.20024289337881573, 1.4166599041256909),
                17.0: (0.23098665490239326, 1.3932074348740047),
                18.0: (0.2777124928693668, 1.426775792099824),
                19.0: (0.2929008042895442, 1.4747510003212119),
                20.0: (0.3620752688172043, 1.7878906008999036),
            },
            "mrp": {
                15.0: (-0.0021444223107569813, 0.9586989846069616),
                16.0: (0.024497463482900377, 1.6357820295692629),
                17.0: (-0.06210995919267675, 1.6115783129601329),
                18.0: (-0.20911066742726755, 1.7337734851287832),
                19.0: (-0.14389812332439678, 1.6338511226429493),
                20.0: (-0.37447311827956997, 1.7742799817329982),
            },
            "mtp": {
                15.0: (0.31768625498007974, 1.5385789105257681),
                16.0: (0.1318500393597481, 2.042439185123313),
                17.0: (0.08599051505459358, 1.7882680591856823),
                18.0: (0.08137706788362807, 1.6678459892972448),
                19.0: (0.02289008042895441, 1.4979308807733835),
                20.0: (0.2165698924731183, 1.176648433236378),
            },
        }

        # Which exponent is closest to requested
        exps = np.fromiter(coefficients["mrr"].keys(), dtype=float)
        e = exps[np.abs(exps - exp).argmin()]

        return coefficients[component][e]

    def moment_rate_triangle(self, half_n=20, amplitude=1.0):
        """
        Function to produce a triangular moment rate function.
        This was the assumed moment rate function used by GCMT after to 2004.

        The triangle is an isosceles triangle with half width equal to the source
        half duration.
        The amplitude and number of points per half duration can be set by the user.
        """

        # Gradient of triangle edge
        m = amplitude / self.hd

        # Moment rate function
        half = np.linspace(0.0, self.hd, half_n + 1) * m
        mrf = np.append(half, np.flip(half)[1:])

        # Time array
        t = np.linspace(0.0, 2.0 * self.hd, 2 * half_n + 1)

        return t, mrf

    def moment_rate_boxcar(self, half_n=20, amplitude=1.0):
        """
        Function to produce a boxcar moment rate function.
        This was the assumed moment rate function used by GCMT prior to 2004.

        The boxcar has half width equal to the source half duration.
        The amplitude and number of points per half duration can be set by the user.
        """

        # Make boxcar
        half = np.append(0.0, np.full(half_n, amplitude))
        mrf = np.append(half, np.flip(half)[1:])

        # Time array
        t = np.linspace(0.0, 2.0 * self.hd, 2 * half_n + 1)

        return t, mrf

    def moment_rate_Gaussian(self, half_n=20, amplitude=1.0):
        """
        Function to produce a Gaussian moment rate function.

        The Gaussian has 3 sigma equal to the source half duration.
        The amplitude and number of points per half duration can be set by the user.
        """

        x = np.linspace(-1.0 * self.hd, 0.0, half_n + 1)
        sigma = self.hd / 3.0
        half = self.Gaussian(x, 0.0, sigma)
        mrf = np.append(half, np.flip(half)[1:])

        # Time array
        t = np.linspace(0.0, 2.0 * self.hd, 2 * half_n + 1)

        return t, mrf

    def Gaussian(self, x, A, B):
        """
        Function returns a Gaussian with mean A and sigma B.
        """

        y = np.exp(-0.5 * ((x - A) / B) ** 2)

        return y

    def skewed_Gaussian(self, x, A, B, C):
        """
        Function returns a skewed Gaussian with skew factor A, mean B and sigma C.
        """

        y = self.Gaussian(x, B, C) * (erf(A * (x - B) / (C * np.sqrt(2.0))) + 1)

        return y
