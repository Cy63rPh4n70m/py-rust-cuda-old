# build rust shared library
cargo build --release
cd target/release
move *.dll ../../test/nnpackage/backend/
move *.exp ../../test/nnpackage/backend/
move *.lib ../../test/nnpackage/backend/
cd ../..