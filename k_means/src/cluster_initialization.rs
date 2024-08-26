pub mod forgy;
pub mod macqueen;
pub mod maximin;
pub mod bradley_fayyad;
pub mod kmeanspp;

#[derive(Debug)]
pub enum CentroidInitStrategy {
    Forgy,
    MacQueen,
    Maximin,
    BradleyFayyad,
    KmeansPP,
    GreedyKmeansPP,
}
