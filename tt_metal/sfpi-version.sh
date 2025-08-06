# set SFPI release version information

sfpi_version=v6.16.1-sfpload
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=a0d931cf2c66df4ef103ed95edc19b03
sfpi_x86_64_Linux_deb_md5=27bc4a95cedadeabf64c8525f3665040
sfpi_x86_64_Linux_rpm_md5=e93661e4f2fb32045b2bbff6aac034c5
