# Shortcuts for the deck. Run from the project root.
#
#   make test          run the math tests
#   make render-low    render all sections fast (480p) for checking
#   make render-fast   render all sections at 1080p, 30 fps (about 2x faster than render)
#   make render        render all sections at 1080p, 60 fps for presenting
#   make s2 QUALITY=h  render one section (s1..s4) at a chosen quality (l or h);
#                      add FPS=30 for the faster 30 fps version
#
# Every render shows a progress bar; Manim's full output goes to media/render.log.
#   make present       open the presenter
#   make html          export a single self-contained slides.html
#   make notes         rebuild notes/speaker_notes.md from the scenes

VENV ?= .venv
BIN := $(abspath $(VENV)/bin)
# MacTeX installs here; adding it means LaTeX works even in a fresh shell.
export PATH := /Library/TeX/texbin:$(PATH)
# Tools are called by quoted full path: this folder's path contains a space,
# which breaks make's own PATH lookup.
PYTHON := "$(BIN)/python"
SLIDES := "$(BIN)/manim-slides"

QUALITY ?= l
FPS ?=
RENDER = $(PYTHON) tools/render.py --quality=$(QUALITY) $(if $(FPS),--fps $(FPS))
BONUS := $(shell $(PYTHON) -c "import config; print(int(config.ENABLE_3D_BONUS))")
SCENES := S1Problem S2Solution S3Sampler $(if $(filter 1,$(BONUS)),S3bThreeD) S4WrapUp

.PHONY: test lint render-low render-fast render s1 s2 s3 s3b s4 present html notes clean

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

s1:
	$(RENDER) s1
s2:
	$(RENDER) s2
s3:
	$(RENDER) s3
s3b:
	$(RENDER) s3b
s4:
	$(RENDER) s4

render-low:
	$(PYTHON) tools/render.py --quality=l

render-fast:
	$(PYTHON) tools/render.py --quality=h --fps 30

render:
	$(PYTHON) tools/render.py --quality=h

present:
	$(SLIDES) present $(SCENES)

html:
	$(SLIDES) convert $(SCENES) slides.html --one-file

notes:
	$(PYTHON) notes/build_notes.py

clean:
	rm -rf media slides slides.html
