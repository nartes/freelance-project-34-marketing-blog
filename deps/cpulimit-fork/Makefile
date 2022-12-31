default: build

build:
	cd src && $(MAKE)
	cd tests && $(MAKE)

pacman: build
	cd pacman-package && makepkg -e -f
