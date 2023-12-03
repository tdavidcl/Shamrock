cd doxygen
doxygen dox.conf

cd ../shamrock-doc
mkdir bin
curl -sSL https://github.com/rust-lang/mdBook/releases/download/v0.4.28/mdbook-v0.4.28-x86_64-unknown-linux-gnu.tar.gz | tar -xz --directory=bin
bin/mdbook build

cd ../mkdocs
mkdocs build

cd ..


rm -rf _build
mkdir _build
cd _build


mkdir mdbook
mkdir doxygen
mkdir mkdocs

cp -r ../shamrock-doc/book/* mdbook
cp -r ../doxygen/html/* doxygen
cp -r ../mkdocs/site/* mkdocs

cp ../tmpindex.html index.html