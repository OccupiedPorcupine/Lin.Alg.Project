# Shortcuts for the deck. Run from the project root.
#
#   make test          run the math tests
#   make render-low    render all sections fast (480p) for checking
#   make render        render all sections at 1080p for presenting
#   make s2 QUALITY=h  render one section (s1..s4) at a chosen quality (l or h)
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

# Use --quality=<x>, not -q<x>: manim-slides reads the "h" in "-qh" as --help.
QUALITY ?= l
Q = --quality=$(QUALITY)
BONUS := $(shell $(PYTHON) -c "import config; print(int(config.ENABLE_3D_BONUS))")
SCENES := S1Problem S2Solution S3Demo $(if $(filter 1,$(BONUS)),S3bThreeD) S4WrapUp

.PHONY: test lint render-low render s1 s2 s3 s3b s4 present html notes clean

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

s1:
	$(SLIDES) render $(Q) scenes/s1_problem.py S1Problem
s2:
	$(SLIDES) render $(Q) scenes/s2_solution.py S2Solution
s3:
	$(SLIDES) render $(Q) scenes/s3_demo.py S3Demo
s3b:
	$(SLIDES) render $(Q) scenes/s3_demo.py S3bThreeD
s4:
	$(SLIDES) render $(Q) scenes/s4_wrapup.py S4WrapUp

render-low:
	$(MAKE) s1 s2 s3 $(if $(filter 1,$(BONUS)),s3b) s4 QUALITY=l

render:
	$(MAKE) s1 s2 s3 $(if $(filter 1,$(BONUS)),s3b) s4 QUALITY=h

present:
	$(SLIDES) present $(SCENES)

html:
	$(SLIDES) convert $(SCENES) slides.html --one-file

notes:
	$(PYTHON) notes/build_notes.py

clean:
	rm -rf media slides slides.html
