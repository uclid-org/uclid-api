from uclid.builder import (
    DeclTypes,
    UclidArraySelect,
    UclidAssignStmt,
    UclidAssumeStmt,
    UclidBMCCommand,
    UclidControlBlock,
    UclidHavocStmt,
    UclidInitBlock,
    UclidLiteral,
    UclidModule,
    UclidNextBlock,
    UclidOpExpr,
    UclidPrintCexJSONCommand,
    UclidPrintResultsCommand,
)
from uclid.builder_sugar import Uadd, UInt

__author__ = "Adwait Godbole"
__copyright__ = "Adwait Godbole"
__license__ = "MIT"


def test_uadd():
    assert Uadd([1, 2]).__inject__() == "(1) + (2)"


def test_add():
    m = UclidModule("main")
    a = m.mkVar("a", UInt)
    b = m.mkVar("b", UInt)
    c = m.mkVar("c", UInt)
    init = UclidInitBlock(
        [
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b),
        ]
    )
    m.setInit(init)
    assert "var c : integer;" in m.__inject__()


def test_uninterpreted_type():
    m = UclidModule("main")
    t = m.mkUninterpretedType("mytype")
    a = m.mkVar("a", UInt)
    b = m.mkVar("b", UInt)
    c = m.mkVar("c", t)
    init = UclidInitBlock(
        [
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b),
        ]
    )
    m.setInit(init)
    assert "var c : mytype;" in m.__inject__()


def test_array_type():
    m = UclidModule("main")
    s = m.mkArrayType("myarr", UInt, UInt)
    a = m.mkVar("a", UInt)
    b = m.mkVar("b", UInt)
    c = m.mkVar("c", s)
    init = UclidInitBlock(
        [
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b),
        ]
    )
    m.setInit(init)
    assert "var c : myarr;" in m.__inject__()


def test_const():
    m = UclidModule("main")
    a = m.mkVar("a", UInt)
    b = m.mkVar("b", UInt)
    c = m.mkVar("c", UInt)
    d = m.mkConst("d", UInt, UclidLiteral(3))
    init = UclidInitBlock(
        [
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b),
            UclidAssignStmt(d, UclidLiteral(3)),
        ]
    )
    m.setInit(init)
    assert "const d : integer = 3;" in m.__inject__()


def test_big():
    m1 = UclidModule("subm")
    m1.mkVar("a", UInt)

    t1 = m1.mkUninterpretedType("mytype")
    t2 = m1.mkArrayType("arr_t", UInt, UInt)
    v1 = m1.mkVar("v1", t1)
    v2 = m1.mkVar("v2", t2)
    init1 = UclidInitBlock(
        [
            UclidAssignStmt(UclidArraySelect(v2, [UclidLiteral(1)]), UclidLiteral(1)),
            UclidHavocStmt(v1),
        ]
    )
    m1.setInit(init1)

    m2 = UclidModule("main")
    a = m2.mkVar("a", UInt)
    b = m2.mkVar("b", UInt)

    m2.mkImport(DeclTypes.TYPE, "t1", m1, "mytype")
    m2.setInit(
        UclidInitBlock(
            [
                UclidAssignStmt(a, UclidLiteral(1)),
                UclidAssignStmt(b, UclidLiteral(2)),
                UclidAssumeStmt(UclidOpExpr("eq", [a, b])),
            ]
        )
    )
    m2.setNext(
        UclidNextBlock(
            [
                UclidAssignStmt(a.p(), UclidOpExpr("add", [a, b])),
                UclidAssignStmt(b.p(), UclidOpExpr("add", [a, UclidLiteral("1")])),
            ]
        )
    )
    m2.setControl(
        UclidControlBlock(
            [
                UclidBMCCommand("v", 10),
                UclidPrintCexJSONCommand("v"),
                UclidPrintResultsCommand(),
            ]
        )
    )

    s1 = m1.__inject__()
    s2 = m2.__inject__()
    assert "type mytype;" in s1
    assert "type arr_t = [integer]integer;" in s1
    assert "var v2 : arr_t;" in s1
    assert "type t1 = subm.mytype;" in s2
