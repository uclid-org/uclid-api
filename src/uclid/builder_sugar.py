

from .builder import UclidContext, UclidBooleanType, UclidIntegerType, PortType, UclidOpExpr, UclidCheckCommand, UclidPrintResultsCommand

# Variable creation sugaring
def mkUclidVar(varname, typ, porttype):
    UclidContext.curr_module.mkVar(varname, typ, porttype)    
def mkUclidVars(varnames : list, typ, porttype):
    return [UclidContext.curr_module.mkVar(varname, typ, porttype) for varname in varnames]
def mkUclidIntVar(varname : str, porttype=PortType.var):
    return UclidContext.curr_module.mkIntegerVar(varname, porttype)
def mkUclidIntVars(varnames : list, porttype=PortType.var):
    return [UclidContext.curr_module.mkIntegerVar(varname, porttype) for varname in varnames]
def mkUclidBoolVar(varname : str, porttype=PortType.var):
    return UclidContext.curr_module.mkBooleanVar(varname, porttype)
def mkUclidBoolVars(varnames : list, porttype=PortType.var):
    return [UclidContext.curr_module.mkBooleanVar(varname, porttype) for varname in varnames]
def mkUclidBVVar(varname : str, width : int, porttype=PortType.var):
    return UclidContext.curr_module.mkBVVar(varname, width, porttype)
    
# Expression sugaring
def Uadd(args):
    return UclidOpExpr("add", args)
def Usub(args):
    return UclidOpExpr("sub", args)
def Uumin(args):
    return UclidOpExpr("umin", args)
def Ugt(args):
    return UclidOpExpr("gt", args)
def Ugte(args):
    return UclidOpExpr("gte", args)
def Ult(args):
    return UclidOpExpr("lt", args)
def Ulte(args):
    return UclidOpExpr("lte", args)
def Ueq(args):
    return UclidOpExpr("eq", args)
def Uneq(args):
    return UclidOpExpr("neq", args)
def Unot(args):
    return UclidOpExpr("not", args)
def Uxor(args):
    return UclidOpExpr("xor", args)
def Uand(args):
    return UclidOpExpr("and", args)
def Uor(args):
    return UclidOpExpr("or", args)
def Uimplies(args):
    return UclidOpExpr("implies", args)
def Ubvadd(args):
    return UclidOpExpr("bvadd", args)
def Ubvsub(args):
    return UclidOpExpr("bvsub", args)
def Ubvand(args):
    return UclidOpExpr("bvand", args)
def Ubvor(args):
    return UclidOpExpr("bvor", args)
def Ubvnot(args):
    return UclidOpExpr("bvnot", args)
def Ubvlt(args):
    return UclidOpExpr("bvlt", args)
def Ubvult(args):
    return UclidOpExpr("bvult", args)
def Ubvgt(args):
    return UclidOpExpr("bvgt", args)
def Ubvugt(args):
    return UclidOpExpr("bvugt", args)
def Unext(args):
    return UclidOpExpr("next", args)
def Ueventually(args):
    return UclidOpExpr("eventually", args)
def Ualways(args):
    return UclidOpExpr("always", args)
def Uconcat(args):
    return UclidOpExpr("concat", args)


UBool = UclidBooleanType()
UInt = UclidIntegerType()


CMD_check = UclidCheckCommand()
CMD_print = UclidPrintResultsCommand()
