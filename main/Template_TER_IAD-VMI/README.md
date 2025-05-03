Ce template Latex pour les TER réalisés dans le cadre du Master IAD et du Master VMI a été réalisé en adaptant Sleek Template. La fin de ce fichier est le README original fourni avec Sleek Template.

Rédigez votre rapport dans le fichier main.tex. Si vous renommez le fichier, pensez à modifier la première ligne (target) du Makefile. Pour compiler, vous pouvez utiliser les commandes :
$ make
-> produira le fichier main.pdf
$ make clean
-> "nettoie" le répertoire, en supprimant le document pdf et tous les fichiers liés à la compilation.

La compilation peut aussi être faite manuellement via la série de commandes suivantes (qui correspond à la commande make) :
$ pdflatex main.tex
$ biber main.bcf
$ pdflatex main.tex
$ pdflatex main.tex


# Sleek Template

Sleek Template is a minimal collection of [LaTeX](https://www.latex-project.org/) packages and settings that ease the writing of beautiful documents. While originally meant for theses, it is perfectly suitable for project reports, articles, syntheses, etc. – with a few adjustments, like margins.

It is composed of four separate packages – `sleek`, `sleek-title`, `sleek-theorems` and `sleek-listings` – each of which can be used individually.

For more information about the packages, settings, environments, commands, etc., please refer to the documentation file [`main.pdf`](main.pdf), which also acts as showcase document.

## Overleaf :leaves:

Sleek Template is [Overleaf](https://www.overleaf.com/) ready!

1. Download the repository [archive](https://github.com/francois-rozet/sleek-template/archive/overleaf.zip)
2. On your Overleaf project page, click **New Project** and choose **Upload Project**
3. Drag or select the downloaded archive
4. Rename the project
5. Enjoy :ok_hand:

## Author

* **François Rozet** - [francois-rozet](https://github.com/francois-rozet)
