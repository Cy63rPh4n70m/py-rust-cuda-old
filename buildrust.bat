# build rust shared library
cargo build --release
cd target/release
move *.dll ../../nnpackage/rust_backend/
move *.exp ../../nnpackage/rust_backend/
move *.lib ../../nnpackage/rust_backend/
cd ../..