import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astro_cubes.defaults import colors
from src.utils.formulae import MS_speagle
from src.utils.tables import cristal_dir, ctab, alptab

# --------------------------------------------------------------------------------------------------

results_2sig = np.load(cristal_dir+"stacks/all_sources_2sig_results.npy", allow_pickle=True).item()
results_3sig = np.load(cristal_dir+"stacks/all_sources_3sig_results.npy", allow_pickle=True).item()
results_high_sfr = np.load(cristal_dir+"stacks/mask_high_sfr_vel_res_med_results.npy", allow_pickle=True).item()
davies19 = ascii.read(cristal_dir+"scripts/cristal/literature/davies19.txt")
fluetsch19 = ascii.read(cristal_dir+"scripts/cristal/literature/fluetsch19_all.txt")
swinbank19 = ascii.read(cristal_dir+"scripts/cristal/literature/swinbank19.txt")
ginolfi20 = ascii.read(cristal_dir+"scripts/cristal/literature/ginolfi20.txt")
concas22 = ascii.read(cristal_dir+"scripts/cristal/literature/concas22.txt")
romano22 = ascii.read(cristal_dir+"scripts/cristal/literature/romano22.txt")
chu24 = np.genfromtxt(cristal_dir+"scripts/cristal/literature/chu24.txt",
                      delimiter=',', skip_header=24, names=True, dtype=None,encoding=None)
schroetter24 = ascii.read(cristal_dir+"scripts/cristal/literature/schroetter24.txt")
weldon24 = ascii.read(cristal_dir+"scripts/cristal/literature/weldon24.txt")

fluetsch19 = fluetsch19[(fluetsch19["type"]=="HII")]# | (fluetsch19["type"]=="LINER")]

# --------------------------------------------------------------------------------------------------

# main sequence
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([2e08, 5e11])
ax.set_ylim([2e00, 1e03])

label_size = 13.5

# ALPINE points
alpine_stack = ax.errorbar(10**alptab["logMstar"][alptab["g20_stack?"]], 10**alptab["logSFR_SED"][alptab["g20_stack?"]],
                     fmt="o", ms=7.5, c=colors[1], mfc=colors[1], zorder=8)
alpine_not_stack = ax.errorbar(10**alptab["logMstar"][alptab["g20_stack?"]==False],
                               10**alptab["logSFR_SED"][alptab["g20_stack?"]==False],
                     fmt="o", ms=7.5, mfc="w", c=colors[1], zorder=7)

# CRISTAL points
cristal_stack = ax.errorbar(10**ctab["logMstar"][ctab["in_stack?"]], 10**ctab["logSFR"][ctab["in_stack?"]],
                            fmt="o", c=colors[0], zorder=10)
cristal_not_stack = ax.errorbar(10**ctab["logMstar"][ctab["in_stack?"]==False],
                               10**ctab["logSFR"][ctab["in_stack?"]==False],
                     fmt="o", mfc="w", c=colors[0], zorder=9)

# # label the points with ID
# for i in range(len(ctab)):
#     if ctab["CRISTAL_ID"][i] in ["C04","C05","C07","C08","C09","C11","C12","C13","C14","C15","C16","C20","C21"]:
#         continue
#     ax.annotate(ctab["CRISTAL_ID"][i], (10**ctab["M_star"][i], 10**ctab["SFR_SED"][i]), fontsize=label_size)

# # C04
# ax.text(1e11, 1e02, "C04", fontsize=label_size)
# ax.plot([0.97e11,10**ctab["M_star"][3]], [1e02,10**ctab["SFR_SED"][3]], c="k")
# # C05
# ax.text(1.55e11, 6.7e01, "C05", fontsize=label_size)
# ax.plot([1.5e11,10**ctab["M_star"][4]], [7e01,10**ctab["SFR_SED"][4]], c="k")
# # C07
# ax.text(1e9, 2e02, "C07", fontsize=label_size)
# ax.plot([1.5e09,10**ctab["M_star"][6]], [1.9e02,10**ctab["SFR_SED"][6]], c="k")
# # C08
# ax.text(4e9, 1.77e02, "C08", fontsize=label_size)
# ax.plot([5e09,10**ctab["M_star"][7]], [1.7e02,10**ctab["SFR_SED"][7]], c="k")
# # C09
# ax.text(3.2e10, 1.55e01, "C09", fontsize=label_size)
# ax.plot([3e10,10**ctab["M_star"][8]], [1.7e01,10**ctab["SFR_SED"][8]], c="k")
# # C11
# ax.text(2.3e09, 8.3e01, "C11", fontsize=label_size)
# ax.plot([3e09,10**ctab["M_star"][10]], [8e01,10**ctab["SFR_SED"][10]], c="k")
# # C12
# ax.text(8e09, 5e00, "C12", fontsize=label_size)
# ax.plot([1e10,10**ctab["M_star"][11]], [6e00,10**ctab["SFR_SED"][11]], c="k")
# # C13
# ax.text(3.15e10, 2.35e01, "C13", fontsize=label_size)
# ax.plot([3e10,10**ctab["M_star"][12]], [2.5e01,10**ctab["SFR_SED"][12]], c="k")
# # C14 and C15 (both are effectively the same)
# ax.text(2.7e8, 5.3e01, "C14,C15", fontsize=label_size)
# ax.plot([5e08,10**ctab["M_star"][13]], [5e01,10**ctab["SFR_SED"][13]], c="k")
# # C16
# ax.text(1.4e09, 5.5e01, "C16", fontsize=label_size)
# ax.plot([2e09,10**ctab["M_star"][15]], [5.3e01,10**ctab["SFR_SED"][15]], c="k")
# # C20
# ax.text(1e11, 2.5e01, "C20", fontsize=label_size)
# ax.plot([0.95e11,10**ctab["M_star"][19]], [2.75e01,10**ctab["SFR_SED"][19]], c="k")
# # C21
# ax.text(2e10, 1e01, "C21", fontsize=label_size)
# ax.plot([1.9e10,10**ctab["M_star"][20]], [1.1e01,10**ctab["SFR_SED"][20]], c="k")

# main sequence line (Speagle+14)
X = np.logspace(8, 12, 500)
sfr_ms_z5 = MS_speagle(5, X)*X/1e09
ax.errorbar(X, sfr_ms_z5, c="k", ls="-", alpha=0.5)
speagle_ms = ax.fill_between(X, sfr_ms_z5/10**0.3, sfr_ms_z5*10**0.3, color="grey", alpha=0.25)

# ax.axhline(np.median(10**ctab["logSFR"]), c="k", ls="--", alpha=0.75)
# ax.axvline(np.median(10**ctab["logMstar"]), c="k", ls="--", alpha=0.75)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_\ast$ / M$_\odot$")
ax.set_ylabel(r"SFR / M$_\odot$yr$^{-1}$")
ax.legend(loc=2, handles=[cristal_stack, cristal_not_stack, alpine_stack, alpine_not_stack, speagle_ms],
          labels=["CRISTAL (in stack)", "CRISTAL (not in stack)", "ALPINE (in G20)", "ALPINE (not in G20)", "Speagle+14 MS"])

plt.savefig(cristal_dir+"plots/main_sequence_cristal_alpine.pdf")
plt.close()

# --------------------------------------------------------------------------------------------------

# [CII] SFR vs UV+IR SFR
fig = plt.figure(figsize=(7.5,7.5))
ax = plt.subplot(111)
ax.set_xlim([5,1000])
ax.set_ylim([5,1000])
cristal_points = ax.errorbar(x = 10**ctab["logSFR"], y = ctab["SFR_CII"], fmt="o")
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [ax.get_ylim()[0],ax.get_ylim()[1]], c="k", ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"SFR$_{\rm SED}$ / M$_\odot$yr$^{-1}$")
ax.set_ylabel(r"SFR$_{\rm [CII]}$ / M$_\odot$yr$^{-1}$")
plt.savefig(cristal_dir+"plots/cii_sfr_uvir_sfr.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# L SFR CRISTAL vs ALPINE
fig = plt.figure(figsize=(7.5,7.5))
ax = plt.subplot(111)
ax.set_xlim([5e07,5e09])
ax.set_ylim([5e07,5e09])
cristal_points = ax.errorbar(x = ctab["L_CII"], y = ctab["L_CII_alp"],
                             xerr = ctab["L_CII_err"], yerr = ctab["L_CII_err_alp"], fmt="o")
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [ax.get_ylim()[0],ax.get_ylim()[1]], c="k", ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"L${\rm [CII],CRISTAL}$ / L$_\odot$")
ax.set_ylabel(r"L$_{\rm [CII],ALPINE}$ / L$_\odot$")
plt.savefig(cristal_dir+"plots/l_cii_cristal_alpine.pdf", bbox_inches="tight")
plt.close()


# --------------------------------------------------------------------------------------------------

# mass outflow rate vs SFR
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([1,300])
ax.set_ylim([0.5,300])

# CRISTAL (full sample)
cristal_points = ax.errorbar(x = results_2sig["SFR_med"],
            y = results_2sig["M_out_dot"],
            xerr = [[results_2sig["SFR_err_l"]/2],[results_2sig["SFR_err_u"]/2]],
            yerr = results_2sig["M_out_dot_err"],
            fmt="o", c=colors[0], ms=17.5)
ax.errorbar(x = results_2sig["SFR_med"]*0.5,
            y = results_2sig["M_out_dot"],
            yerr = results_2sig["M_out_dot"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = results_2sig["SFR_med"]*0.5,
            y = results_2sig["M_out_dot"],
            yerr = (results_2sig["M_out_dot"]-results_2sig["M_out_dot_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

# CRISTAL (high-SFR sample)
cristal_points_high_sfr = ax.errorbar(x = results_high_sfr["SFR_med"],
            y = results_high_sfr["M_out_dot"],
            xerr = [[results_high_sfr["SFR_err_l"]/2],[results_high_sfr["SFR_err_u"]/2]],
            yerr = results_high_sfr["M_out_dot_err"],
            fmt="o", c=colors[0], mfc="w", ms=17.5)
ax.errorbar(x = results_high_sfr["SFR_med"]*0.5,
            y = results_high_sfr["M_out_dot"],
            yerr = results_high_sfr["M_out_dot"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = results_high_sfr["SFR_med"]*0.5,
            y = results_high_sfr["M_out_dot"],
            yerr = (results_high_sfr["M_out_dot"]-results_high_sfr["M_out_dot_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

fluetsch19_points = ax.errorbar(x = fluetsch19["SFR"],
            y = fluetsch19["M_out_dot"],
            fmt="o", c=colors[5], ms=8)
swinbank19_points = ax.errorbar(x = swinbank19["SFR"],
            y = swinbank19["M_out_dot"],
            fmt="o", c=colors[1], ms=8)
ginolfi20_points = ax.errorbar(x = ginolfi20["SFR"],
            y = ginolfi20["M_out_dot"],
            fmt="o", c=colors[2], ms=8)
weldon24_points = ax.errorbar(x = 10**weldon24["log_SFR"],
            y = weldon24["M_out_dot"],
            fmt="o", c=colors[3], ms=8)
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [5*ax.get_xlim()[0],5*ax.get_xlim()[1]],
        c="k", ls="--", alpha=0.75)
ax.text(2, 11.5, r"$\eta$ = 5.0", ha="center", va="center", rotation=41, fontsize=15)
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [ax.get_xlim()[0],ax.get_xlim()[1]],
        c="k", ls="--", alpha=0.75)
ax.text(2, 2.3, r"$\eta$ = 1.0", ha="center", va="center", rotation=41, fontsize=15)
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [0.5*ax.get_xlim()[0],0.5*ax.get_xlim()[1]],
        c="k", ls="--", alpha=0.75)
ax.text(3, 1.7, r"$\eta$ = 0.5", ha="center", va="center", rotation=41, fontsize=15)
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [0.1*ax.get_xlim()[0],0.1*ax.get_xlim()[1]],
        c="k", ls="--", alpha=0.75)
ax.text(8, 0.9, r"$\eta$ = 0.1", ha="center", va="center", rotation=41, fontsize=15)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"SFR / M$_\odot$yr$^{-1}$")
ax.set_ylabel(r"$\dot{M}_{\rm out}$ / M$_\odot$yr$^{-1}$")
leg = ax.legend(loc=2, handles=[cristal_points,cristal_points_high_sfr,fluetsch19_points,swinbank19_points,ginolfi20_points,weldon24_points],
                labels=["CRISTAL (All)",r"CRISTAL (High-$\Sigma_{\rm SFR}$)","Fluetsch+19","Swinbank+19","Ginolfi+20","Weldon+24"])
plt.savefig(cristal_dir+"plots/m_out_dot_sfr.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# mass loading factor vs SFR
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([1,300])
ax.set_ylim([0.05,3])

# CRISTAL (full sample)
cristal_points = ax.errorbar(x = results_2sig["SFR_med"],
            y = results_2sig["load_factor"],
            xerr = [[results_2sig["SFR_err_l"]],[results_2sig["SFR_err_u"]]],
            yerr = results_2sig["load_factor_err"],
            fmt="o", c=colors[0])
ax.errorbar(x = results_2sig["SFR_med"]*0.5,
            y = results_2sig["M_out_dot"],
            yerr = results_2sig["M_out_dot"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = results_2sig["SFR_med"]*0.5,
            y = results_2sig["M_out_dot"],
            yerr = (results_2sig["M_out_dot"]-results_2sig["M_out_dot_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

# cristal_points_high_sfr = ax.errorbar(x = results_high_sfr["SFR_med"],
#             y = results_high_sfr["M_out_dot"],
#             xerr = results_high_sfr["SFR_err"],
#             yerr = results_high_sfr["M_out_dot_err"],
#             fmt=".", marker="o", c=colors[0], mfc="w")
swinbank19_points = ax.errorbar(x = swinbank19["SFR"],
            y = swinbank19["load_factor"],
            fmt=".", c=colors[1])
ginolfi20_points = ax.errorbar(x = ginolfi20["SFR"],
            y = ginolfi20["load_factor"],
            fmt=".", c=colors[2])
concas22_points = ax.errorbar(x = 10**concas22["log_SFR"],
            y = 10**concas22["log_mass_load"],
            fmt=".", c=colors[4])
weldon24_points = ax.errorbar(x = 10**weldon24["log_SFR"],
            y = 10**weldon24["log_load_factor_m"],
            fmt=".", c=colors[3])
# ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [ax.get_xlim()[0],ax.get_xlim()[1]],
#         c="k", ls="--", alpha=0.75)
# ax.text(2, 2.3, r"$\eta$ = 1.0", ha="center", va="center", rotation=51, fontsize=15)
# ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [0.5*ax.get_xlim()[0],0.5*ax.get_xlim()[1]],
#         c="k", ls="--", alpha=0.75)
# ax.text(3, 1.7, r"$\eta$ = 0.5", ha="center", va="center", rotation=51, fontsize=15)
# ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [0.1*ax.get_xlim()[0],0.1*ax.get_xlim()[1]],
#         c="k", ls="--", alpha=0.75)
# ax.text(8, 0.9, r"$\eta$ = 0.1", ha="center", va="center", rotation=51, fontsize=15)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"SFR / M$_\odot$yr$^{-1}$")
ax.set_ylabel(r"$\eta_m$")
leg = ax.legend(loc=4, handles=[cristal_points,swinbank19_points,ginolfi20_points,weldon24_points],
                labels=["CRISTAL","Swinbank+19","Ginolfi+20","Weldon+24"])
plt.savefig(cristal_dir+"plots/mass_load_sfr.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# mass loading factor vs sigma_SFR
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([0.03,10])
ax.set_ylim([0.03,5])
X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)

# CRISTAL points (all, 2-sigma sample)
cristal_points = ax.errorbar(x = 1.71,
            y = results_2sig["load_factor"],
            xerr = [[0.96],[2.55]],
            yerr = results_2sig["load_factor_err"],
            fmt="o", c=colors[0])
ax.errorbar(x = 1.71*0.2,
            y = results_2sig["load_factor"],
            yerr = results_2sig["load_factor"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = 1.713*0.2,
            y = results_2sig["load_factor"],
            yerr = (results_2sig["load_factor"]-results_2sig["load_factor_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

# CRISTAL points (high-SFR sample)
cristal_points_high_sfr = ax.errorbar(x = 3.13,
            y = results_high_sfr["load_factor"],
            xerr = [[0.95],[2.81]],
            yerr = results_high_sfr["load_factor_err"],
            fmt="o", c=colors[0], mfc="w")
ax.errorbar(x = 3.13*0.2,
            y = results_high_sfr["load_factor"],
            yerr = results_high_sfr["load_factor"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = 3.13*0.2,
            y = results_high_sfr["load_factor"],
            yerr = (results_high_sfr["load_factor"]-results_high_sfr["load_factor_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

davies19_points = ax.errorbar(x = davies19["Sigma_SFR"],
            y = davies19["eta"],
            fmt=".", c=colors[1])
# duvet = ax.errorbar(x = X, y = 10**(0.17+0.04*np.log10(X)), c="k")
duvet = ax.errorbar(x = chu24["sigma_sfr"], y = chu24["mlf"], fmt=".", c=colors[6])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Sigma_{\rm SFR}$ / M$_\odot$yr$^{-1}$kpc$^{-2}$")
ax.set_ylabel(r"$\eta_m$")
leg = ax.legend(loc=4, handles=[cristal_points,davies19_points, duvet],
                labels=["CRISTAL","Davies+19","DUVET"])
plt.savefig(cristal_dir+"plots/mass_load_sig_sfr.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# v_out vs sigma_SFR
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([1e-03,3e03])
ax.set_ylim([30,3000])
X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)

# CRISTAL points (all, 2-sigma sample)
cristal_points = ax.errorbar(x = 1.71,
            y = results_2sig["v_out"],
            xerr = [[0.96],[2.55]],
            yerr = results_2sig["v_out_err"],
            fmt="o", c=colors[0], zorder=11, ms=15)

# CRISTAL points (high-SFR sample)
cristal_points_high_sfr = ax.errorbar(x = 3.13,
            y = results_high_sfr["v_out"],
            xerr = [[0.95],[2.81]],
            yerr = results_high_sfr["v_out_err"],
            fmt="o", c=colors[0], mfc="w", zorder=10, ms=15, mew=2)

davies19_points = ax.errorbar(x = davies19["Sigma_SFR"],
            y = davies19["v_out"],
            fmt="o", c=colors[1], ms=8)

rhc21_points = ax.errorbar(x = [10**-0.45,10**0.1], y = [590,455], fmt="o", c=colors[3], ms=12)

schroetter24_points = ax.errorbar(x = schroetter24["Sigma_SFR"],
            y = schroetter24["v_out"],
            fmt="o", c=colors[2], ms=8, alpha=0.8)

# duvet = ax.errorbar(x = X, y = 10**(2.44+0.19*np.log10(X)), c="k")
duvet = ax.errorbar(x = chu24["sigma_sfr"], y = chu24["vout_Hbeta"], fmt="o", c="m", ms=8, alpha=0.5, zorder=1)
h15_fit = ax.errorbar(x = X, y = 3296/((X/1307.9)**-0.34 + (X/1307.9)**0.15), c="k", ls="--")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Sigma_{\rm SFR}$ / M$_\odot$yr$^{-1}$kpc$^{-2}$")
ax.set_ylabel(r"$v_{\rm out}$ / kms$^{-1}$")
leg = ax.legend(loc=4, handles=[cristal_points,cristal_points_high_sfr,davies19_points,
                                rhc21_points,schroetter24_points,duvet,h15_fit],
                labels=["CRISTAL (All)",r"CRISTAL (High-$\Sigma_{\rm SFR}$)","Davies+19",
                "Herrera-Camus+21 (HZ4)", "Schroetter+24","DUVET","Heckman+16"],
                fontsize=12)
plt.savefig(cristal_dir+"plots/v_out_sig_sfr.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# Mass loading factor vs Mstar
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([1e08,1e11])
ax.set_ylim([1e-02,1e01])
X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])

# CRISTAL points (all, 2-sigma sample)
cristal_points_2sig = ax.errorbar(x = results_2sig["Mstar_med"],
            y = results_2sig["load_factor"],
            xerr = [[results_2sig["Mstar_err_l"]],[results_2sig["Mstar_err_u"]]],
            yerr = results_2sig["load_factor_err"],
            fmt="o", c=colors[0], ms=17.5, zorder=10)
ax.errorbar(x = results_2sig["Mstar_med"]*0.2,
            y = results_2sig["load_factor"],
            yerr = results_2sig["load_factor"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = results_2sig["Mstar_med"]*0.2,
            y = results_2sig["load_factor"],
            yerr = (results_2sig["load_factor"]-results_2sig["load_factor_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

# CRISTAL points (high-SFR sample)
cristal_points_high_sfr = ax.errorbar(x = results_high_sfr["Mstar_med"],
            y = results_high_sfr["load_factor"],
            xerr = [[results_high_sfr["Mstar_err_l"]],[results_high_sfr["Mstar_err_u"]]],
            yerr = results_high_sfr["load_factor_err"],
            fmt="o", c=colors[0], mfc="w", ms=17.5, zorder=10)
ax.errorbar(x = results_high_sfr["Mstar_med"]*0.2,
            y = results_high_sfr["load_factor"],
            yerr = results_high_sfr["load_factor"]*2,
            lolims=True, c=colors[0], alpha=0.8)
ax.errorbar(x = results_high_sfr["Mstar_med"]*0.2,
            y = results_high_sfr["load_factor"],
            yerr = (results_high_sfr["load_factor"]-results_high_sfr["load_factor_wings"]),
            uplims=True, c=colors[0], alpha=0.8)

# Swinbank+19 points
swinbank19_points = ax.errorbar(x = swinbank19["Mstar"],
            y = swinbank19["load_factor"],
            fmt="o", c=colors[1], ms=8, zorder=5)
# Concas+22 points
# concas22_points = ax.errorbar(x = 10**concas22["log_Mstar"],
#             y = 10**concas22["log_mass_load"],
#             fmt=".", c=colors[5])
# Romano+22 points
romano22_points = ax.errorbar(x = 10**romano22["log_Mstar"],
            y = romano22["load_factor"],
            fmt="o", c=colors[2], ms=8, zorder=5)
# Weldon+24 points
weldon24_points = ax.errorbar(x = 10**weldon24["log_Mstar"],
            y = 10**weldon24["log_load_factor_m"],
            fmt="o", c=colors[3], ms=8, zorder=5)

# Weldon+24 best fit
# weldon24_fit = ax.errorbar(X, 10**(3.95-0.45*np.log10(X)), fmt="-", c=colors[3])
# Muratov+15 (FIRE simulation)
muratov15fire = ax.errorbar(X, 3.6*(X/1e10)**-0.35, fmt="--", c="k", alpha=0.5)
# Pandya+21 (FIRE2 simulation)
pandya21fire = ax.errorbar(X, 10**4.3*(X)**-0.43, fmt="--", c="k")

# Add labels and legend, save
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_\ast$ / M$_\odot$")
ax.set_ylabel(r"$\eta_m$")
leg = ax.legend(loc=3, handles=[cristal_points_2sig,cristal_points_high_sfr,swinbank19_points,
                                romano22_points, weldon24_points,muratov15fire,pandya21fire],
                labels=["CRISTAL (All)",r"CRISTAL (High-$\Sigma_{\rm SFR}$)","Swinbank+19","Romano+23","Weldon+24",
                        "Muratov+15 (FIRE-1)","Pandya+21 (FIRE-2)"])
plt.savefig(cristal_dir+"plots/mass_load_mstar.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# energy rate vs SFR
fig = plt.figure()
ax = plt.subplot(111)
# ax.set_xlim([1,300])
# ax.set_ylim([0.05,3])
cristal_points = ax.errorbar(x = results_2sig["SFR_med"],
            y = results_2sig["E_out_dot"],
            xerr = [[results_2sig["SFR_err_l"]],[results_2sig["SFR_err_u"]]],
            yerr = results_2sig["E_out_dot_err"],
            fmt="o", c=colors[0])
# cristal_points_high_sfr = ax.errorbar(x = results_high_sfr["SFR_med"],
#             y = results_high_sfr["M_out_dot"],
#             xerr = results_high_sfr["SFR_err"],
#             yerr = results_high_sfr["M_out_dot_err"],
#             fmt=".", marker="o", c=colors[0], mfc="w")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"SFR / M$_\odot$yr$^{-1}$")
ax.set_ylabel(r"$\dot{E_{\rm out}}$")
# leg = ax.legend(loc=4, handles=[cristal_points,swinbank19_points,ginolfi20_points,weldon24_points],
#                 labels=["CRISTAL","Swinbank+19","Ginolfi+20","Weldon+24"])
plt.savefig(cristal_dir+"plots/E_out_dot_sfr.pdf", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------

# Ib/In vs SFR
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim([1,300])
ax.set_ylim([0.05,3])
cristal_points = ax.errorbar(x = results_2sig["SFR_med"],
            y = results_2sig["out_frac"],
            xerr = [[results_2sig["SFR_err_l"]/2],[results_2sig["SFR_err_u"]/2]],
            yerr = results_2sig["out_frac_err"],
            fmt="o", c=colors[0])
cristal_points_high_sfr = ax.errorbar(x = results_high_sfr["SFR_med"],
            y = results_high_sfr["out_frac"],
            xerr = [[results_high_sfr["SFR_err_l"]/2],[results_high_sfr["SFR_err_u"]/2]],
            yerr = results_high_sfr["out_frac_err"],
            fmt="o", c=colors[0], mfc="w")
# swinbank19_points = ax.errorbar(x = swinbank19["SFR"],
#             y = swinbank19["load_factor"],
#             fmt=".", c=colors[1])
# ginolfi20_points = ax.errorbar(x = ginolfi20["SFR"],
#             y = ginolfi20["load_factor"],
#             fmt=".", c=colors[2])
# weldon24_points = ax.errorbar(x = 10**weldon24["log_SFR"],
#             y = 10**weldon24["log_load_factor_m"],
#             fmt=".", c=colors[3])
# ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [ax.get_xlim()[0],ax.get_xlim()[1]],
#         c="k", ls="--", alpha=0.75)
# ax.text(2, 2.3, r"$\eta$ = 1.0", ha="center", va="center", rotation=51, fontsize=15)
# ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [0.5*ax.get_xlim()[0],0.5*ax.get_xlim()[1]],
#         c="k", ls="--", alpha=0.75)
# ax.text(3, 1.7, r"$\eta$ = 0.5", ha="center", va="center", rotation=51, fontsize=15)
# ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [0.1*ax.get_xlim()[0],0.1*ax.get_xlim()[1]],
#         c="k", ls="--", alpha=0.75)
# ax.text(8, 0.9, r"$\eta$ = 0.1", ha="center", va="center", rotation=51, fontsize=15)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"SFR / M$_\odot$yr$^{-1}$")
ax.set_ylabel(r"$I_{\rm broad}/I_{\rm narrow}$")
# leg = ax.legend(loc=4, handles=[cristal_points,swinbank19_points,ginolfi20_points,weldon24_points],
#                 labels=["CRISTAL","Swinbank+19","Ginolfi+20","Weldon+24"])
plt.savefig(cristal_dir+"plots/out_frac_sfr.pdf", bbox_inches="tight")
plt.close()