# build rust shared library
cargo build --release
cd target/release
move *.dll ../../nnpackage/backend/
move *.exp ../../nnpackage/backend/
move *.lib ../../nnpackage/backend/
cd ../..