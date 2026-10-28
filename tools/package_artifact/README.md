# Package Artifact Tool

This tool assembles and verifies the Linux OpenCV package artifact consumed by
Terlan package distribution. It copies only generated package sources,
release-mode runtime executables, the selected OpenCV shared-library closure,
and license files into a relocatable directory. It then emits deterministic
payload checksums and a `.tar.zst` archive.

Run it through the package-owned Make targets. The command-line interface is an
internal build contract, not a user installation interface.

The release workflow also uses this tool to resolve reviewed compiler and
OpenCV revisions, require a clean stable package tag, verify the external
archive checksum before extraction, and bind the archive manifest to the
release commit and target. Development versions and dirty source trees cannot
produce a valid release plan.
