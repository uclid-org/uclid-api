import logging
import textwrap

from enum import Enum
from typing import List, Dict

__author__ = "Adwait Godbole"
__copyright__ = "Adwait Godbole"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

class UclidContext():
    # dict: str (modulename) -> UclidModule
    modules             = {}
    # name of current module
    curr_module_name    = "main"
    # current module context
    curr_module         = None

    @staticmethod
    def clearAll():
        """
            Deletes all context information (restart).
        """
        UclidContext.curr_module_name   = "main"
        UclidContext.curr_module = UclidModule("main")
        UclidContext.modules = { "main" : UclidModule("main") }

    @staticmethod
    def setContext(module_name):
        UclidContext.curr_module_name = module_name
        if module_name not in UclidContext.modules:
            UclidContext.modules[module_name]  = UclidModule(module_name)

    @staticmethod
    def __inject__():
        acc = ""
        for modulename in UclidContext.modules:
            acc += UclidContext.modules[modulename].__inject__()
        return acc

class UclidElement():
    def __init__(self) -> None:
        pass
    def __inject__(self) -> str:
        raise NotImplementedError

class PortType(Enum):
    var     = 0
    input   = 1
    output  = 2
# ==============================================================================
# Uclid Declarations
# ==============================================================================
class DeclTypes(Enum):
    VAR = 0
    FUNCTION = 1
    TYPE = 2
    INSTANCE = 3
    SYNTHFUN = 4
    DEFINE = 5
    CONST = 6
    PROCEDURE = 7
    CONSTRAINTS = 8

# Base class for (all sorts of) uclid declarations
class UclidDecl(UclidElement):
    def __init__(self, decltype) -> None:
        super().__init__()
        self.decltype = decltype
    def __inject__(self) -> str:
        if self.decltype == DeclTypes.VAR:
            return "{} {} : {};".format(
                self.porttype.name, self.name, self.__declstring__
            )
        elif self.decltype == DeclTypes.TYPE:
            if self.__declstring__ == "":
                return "type {};".format(self.name)
            return "type {} = {};".format(self.name, self.__declstring__)
        elif self.decltype == DeclTypes.INSTANCE:
            return "instance {} : {};".format(self.name, self.__declstring__)
        elif self.decltype == DeclTypes.CONST:
            return "const {} : {};".format(self.name, self.__declstring__)
        elif self.decltype == DeclTypes.DEFINE:
            return self.__declstring__
        elif self.decltype == DeclTypes.FUNCTION:
            return self.__declstring__
        elif self.decltype == DeclTypes.PROCEDURE:
            return self.__declstring__
        elif self.decltype == DeclTypes.CONSTRAINTS:
            return self.__declstring__
        else:
            _logger.error(f"Declaration for decltype {self.decltype} is not permitted")
            exit(1)
    @property
    def __declstring__(self) -> str:
        raise NotImplementedError

class UclidTypeDecl(UclidDecl):
    def __init__(self, name: str, typexp=None) -> None:
        super().__init__(DeclTypes.TYPE)
        self.name = name
        self.typexp = typexp
    @property
    def __declstring__(self):
        return "" if self.typexp is None else self.typexp.__inject__()

class UclidVarDecl(UclidDecl):
    def __init__(self, name, typ, porttype=PortType.var):
        super().__init__(DeclTypes.VAR)
        self.name = name
        self.typ = typ
        self.porttype = porttype
    @property
    def __declstring__(self):
        return self.typ.__inject__()

class UclidConstDecl(UclidDecl):
    def __init__(self, name, typ, val = None):
        super().__init__(DeclTypes.CONST)
        self.name = name
        self.typ = typ
        self.val = val
    @property
    def __declstring__(self):
        if self.val is None:
            return self.typ.__inject__()
        return f"{self.typ.__inject__()} = {self.val.__inject__()}"

class UclidDefineDecl(UclidDecl):
    def __init__(self, name: str, functionsig, body) -> None:
        super().__init__(DeclTypes.DEFINE)
        self.name = name
        self.functionsig = functionsig
        self.body = body
    @property
    def __declstring__(self) -> str:
        return "\tdefine {}{} = {};".format(
            self.name, self.functionsig.__inject__(), self.body.__inject__()
        )
class UclidFunctionDecl(UclidDecl):
    def __init__(self, name: str, functionsig) -> None:
        super().__init__(DeclTypes.FUNCTION)
        self.name = name
        self.functionsig = functionsig
    @property
    def __declstring__(self) -> str:
        return "\tfunction {}{};".format(
            self.name, self.functionsig.__inject__()
        )

class UclidProcedureDecl(UclidDecl):
    def __init__(self, name: str, proceduresig, body):
        super().__init__(DeclTypes.PROCEDURE)
        self.name = name
        self.proceduresig = proceduresig
        self.body = body
    @property
    def __declstring__(self) -> str:
        return """procedure {} 
    {}
{{
{} 
}}
    """.format(
            self.name, 
            self.proceduresig.__inject__(), 
            textwrap.indent(self.body.__inject__(), '\t')
        )


class UclidInstanceDecl(UclidDecl):
    def __init__(self, instancename, module, argmap):
        super().__init__(DeclTypes.INSTANCE)
        self.name = instancename
        self.module = module
        self.argmap = argmap
    @property
    def __declstring__(self):
        if self.modulename not in UclidContext.modules:
            _logger.error("Module {} not found in UclidContext.modules".format(self.modulename))
            _logger.debug("Available modules: {}".format(UclidContext.modules.keys()))
            exit(1)
        argmapstr = ', '.join([
            "{} : ({})".format(port.name, self.argmap[port.name].__inject__())
            for port in self.module.ip_var_decls + self.module.op_var_decls
        ])
        return "{}({})".format(self.module.name, argmapstr)

class UclidRawInstanceDecl(UclidDecl):
    """ Raw (external) module instances in Uclid """
    def __init__(self, instancename, module, argmap):
        super().__init__(DeclTypes.INSTANCE)
        self.name = instancename
        self.modname = module.name if isinstance(module, UclidModule) else module
        self.argmap = argmap
    @property
    def __declstring__(self):
        argmapstr = ', '.join([
            "{} : ({})".format(portname, self.argmap[portname].__inject__()) 
            for portname in self.argmap
        ])
        return "{}({})".format(self.modname, argmapstr)

class UclidImportDecl(UclidDecl):
    def __init__(self, decltype, name, modulename, refname) -> None:
        """Import objects from a module

        Args:
            decltype (DeclTypes): type of objects to import
            name (str): Name of imported object in current module
            modulename (Module/str): Module (or its name) from which to import
            refname (str): Name of object in the module
        """
        super().__init__(decltype)
        self.name = name
        self.modulename = modulename if isinstance(modulename, str) else modulename.name
        self.refname = refname
    @property
    def __declstring__(self):
        return f"{self.modulename}.{self.refname}"
class UclidWildcardImportDecl(UclidImportDecl):
    def __init__(self, decltype, modulename) -> None:
        super().__init__(decltype, "*", modulename, "*")

# ==============================================================================
# Uclid Spec (assertions)
# ==============================================================================
class UclidSpecDecl(UclidDecl):
    def __init__(self, name: str, body, is_ltl=False) -> None:
        """Specification declaration"""
        super().__init__(DeclTypes.CONSTRAINTS)
        self.name = name
        self.body = body
        self.is_ltl = is_ltl
    @property
    def __declstring__(self) -> str:
        if self.name != "":
            if not self.is_ltl:
                return "property {} : {};\n".format(
                    self.name, self.body.__inject__()
                )
            return "property[LTL] {} : {};\n".format(
                    self.name, self.body.__inject__()
                )
        else:
            if not self.is_ltl:
                return "property {};\n".format(self.body.__inject__())
            return "property[LTL] {};\n".format(self.body.__inject__())

# ==============================================================================
# Uclid Axiom (assumptions)
# ==============================================================================
class UclidAxiomDecl(UclidDecl):
    def __init__(self, name: str, body) -> None:
        super().__init__(DeclTypes.CONSTRAINTS)
        self.name = name
        self.body = body
    def __declstring__(self) -> str:
        if self.name != "":
            return "axiom {} : ({});\n".format(self.name, self.body.__inject__())
        else:
            return "axiom ({});".format(self.body.__inject__())

# ==============================================================================
# Uclid Init
# ==============================================================================
class UclidInitBlock(UclidElement):
    def __init__(self, block) -> None:
        super().__init__()
        if isinstance(block, UclidBlockStmt):
            self.block = block
        elif isinstance(block, list):
            self.block = UclidBlockStmt(block)
        elif isinstance(block, UclidStmt):
            self.block = UclidBlockStmt([block])
        else:
            _logger.error("Unsupported type {} of block in UclidInitBlock".format(
                type(block)
            ))

    def __inject__(self) -> str:
        return textwrap.dedent("""
            init {{
            {}
            }}""").format(textwrap.indent(self.block.__inject__().strip(), '\t'))


# ==============================================================================
# Uclid Next
# ==============================================================================
class UclidNextBlock(UclidElement):
    def __init__(self, block) -> None:
        super().__init__()
        if isinstance(block, UclidBlockStmt):
            self.block = block
        elif isinstance(block, list):
            self.block = UclidBlockStmt(block)
        elif isinstance(block, UclidStmt):
            self.block = UclidBlockStmt([block])
        else:
            _logger.error("Unsupported type {} of block in UclidNextBlock".format(
                type(block)
            ))

    def __inject__(self) -> str:
        return textwrap.dedent("""
            next {{
            {}
            }}""").format(textwrap.indent(self.block.__inject__().strip(), '\t'))

# ==============================================================================
# Uclid Types
# ==============================================================================
class UclidType(UclidElement):
    def __init__(self, typestring):
        self.typestring = typestring
    def __inject__(self) -> str:
        return self.typestring

class UclidBooleanType(UclidType):
    def __init__(self):
        super().__init__("boolean")
class UclidIntegerType(UclidType):
    def __init__(self):
        super().__init__("integer")
class UclidBVType(UclidType):
    def __init__(self, width):
        super().__init__("bv{}".format(width))
        self.width = width
class UclidArrayType(UclidType):
    def __init__(self, itype, etype):
        super().__init__("[{}]{}".format(itype.__inject__(), etype.__inject__()))
        self.indextype = itype
        self.elemtype = etype
class UclidEnumType(UclidType):
    def __init__(self, members):
        super().__init__("enum {{ {} }}".format(", ".join(members)))
        self.members = members
class UclidSynonymType(UclidType):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
class UclidUninterpretedType(UclidType):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

# ==============================================================================
# Uclid Define macros and Functions
# ==============================================================================
class UclidFunctionSig(UclidElement):
    def __init__(self, inputs: list, outtype: UclidType) -> None:
        """Uclid function signature

        Args:
            inputs (List[(UclidLiteral|str, UclidType)]): List of input arguments
            outtype (UclidType): Output type
        """
        super().__init__()
        self.inputs = inputs
        self.outtype = outtype
    def __inject__(self) -> str:
        input_sig = ', '.join(["{} : {}".format(
            i[0] if isinstance(i[0], str) else i[0].__inject__(),
            i[1].__inject__()
        ) for i in self.inputs])
        return "({}) : {}".format(input_sig, self.outtype.__inject__())
class UclidDefine(UclidElement):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
    def __inject__(self) -> str:
        return self.name
class UclidFunction(UclidElement):
    def __init__(self, name) -> None:
        """Uclid uninterpreted function"""
        super().__init__()
        self.name = name
    def __inject__(self) -> str:
        return self.name
# ==============================================================================
# Uclid Procedures
# ==============================================================================
class UclidProcedureSig(UclidElement):
    # ip_args are pairs of str, Uclidtype elements
    def __init__(self, inputs, modifies = None, returns = None, requires = None, ensures = None) -> None:
        """Procedure signature

        Args:
            inputs (List[(UclidLiteral|str, UclidType)]): List of (typed) input arguments
            modifies (List[UclidLiteral], optional): List of modified variables. Defaults to None.
            returns (List[(UclidLiteral|str, UclidType)], optional): List of returned variables. Defaults to None.
            requires (UclidExpr, optional): Input/initial assumptions in procedural verification. Defaults to None.
            ensures (UclidExpr, optional): Output/final guarantees in procedural verification. Defaults to None.
        """
        super().__init__()
        self.inputs = inputs
        self.modifies = modifies
        self.returns = returns
        self.requires = requires
        self.ensures = ensures
    def __inject__(self) -> str:
        input_str = ', '.join(["{} : {}".format(
            i[0] if isinstance(i[0], str) else i[0].lit, 
            i[1].__inject__()
        ) for i in self.inputs])
        return_str = "\n\treturns ({})".format(', '.join(["{} : {}".format(
            i[0] if isinstance(i[0], str) else i[0].lit, 
            i[1].__inject__()) for i in self.returns])
        ) if self.returns is not None else ''
        modify_str = "\n\tmodifies {};".format(
            ', '.join([sig.__inject__() for sig in self.modifies])
        ) if self.modifies is not None else ''
        requires_str = "\nrequires ({});".format(
            self.requires.__inject__()
        ) if self.requires is not None else ''
        ensures_str = "\nensures ({});".format(
            self.ensures.__inject__()
        ) if self.ensures is not None else ''
        return "({}){}{}{}{}".format(input_str, modify_str, return_str, requires_str, ensures_str)
class UclidProcedure(UclidElement):
    def __init__(self, name):
        super().__init__()
        self.name = name
    def __inject__(self) -> str:
        return self.name
# ==============================================================================
# Uclid sub-module instances
# ==============================================================================
class UclidInstance(UclidElement):
    """Module instances in Uclid"""
    def __init__(self, instancename) -> None:
        super().__init__()
        self.instancename = instancename
    def __inject__(self) -> str:
        return self.instancename
# ==============================================================================
# Uclid Expressions
# ==============================================================================
class Operators:
    """
    Comprehensive operator list
    """
    OpMapping = {
        "add"   : ("+",     2),
        "sub"   : ("-",     2),
        "umin"  : ("-",     1),
        "gt"    : (">",     2),
        "gte"   : (">=",    2),
        "lt"    : ("<",     2),
        "lte"   : ("<=",    2),
        "eq"    : ("==",    2),
        "neq"   : ("!=",    2),
        "not"   : ("!",     1),
        "xor"   : ("^",     2),
        "and"   : ("&&",   -1),
        "or"    : ("||",   -1),
        "implies"       : ("==>",   2),
        "bvadd" : ("+",     2),
        "bvsub" : ("-",     2),
        "bvand" : ("&",     2),
        "bvor"  : ("|",     2),
        "bvnot" : ("~",     1),
        "bvlt"  : ("<",     2),
        "bvult" : ("<_u",   2),
        "bvgt"  : (">",     2),
        "bvugt" : (">_u",   2),
        "next"          : ("X",     1),
        "eventually"    : ("F",     1),
        "always"        : ("G",     1),
        "concat" : ("++", -1)
    }

    def __init__(self, op_) -> None:
        self.op = op_
    def __inject__(self) -> str:
        return Operators.OpMapping[self.op][0]

# Base class for Uclid expressions
class UclidExpr(UclidElement):
    def __inject__(self) -> str:
        raise NotImplementedError
    def __make_op__(self, op, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr(op, [self, other])
        elif isinstance(other, int):
            return UclidOpExpr(op, [self, UclidLiteral(str(other))])
        elif isinstance(other, str):
            return UclidOpExpr(op, [self, UclidLiteral(other)])
        else:
            _logger.error("Unsupported types for operation {}: {} and {}".format(
                op, 'UclidExpr', type(other)))

    def __eq__(self, other):
        return self.__make_op__("eq", other)
    def __ne__(self, other):
        return self.__make_op__("neq", other)
    def __add__(self, other):
        return self.__make_op__("add", other)
    def __sub__(self, other):
        return self.__make_op__("sub", other)
    def __mul__(self, other):
        return self.__make_op__("mul", other)
    def __invert__(self):
        return UclidOpExpr("not", [self])
    def __and__(self, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr("and", [self, other])
        else:
            _logger.error("Unsupported types for operation {}: {} and {}".format(
                '&', 'UclidExpr', type(other)))


class UclidOpExpr(UclidExpr):
    """
        Generic Uclid operator expression
    """
    def __init__(self, op, children) -> None:
        super().__init__()
        self.op = op
        self.children = [
            UclidLiteral(str(child)) if isinstance(child, int) else child  
            for child in children
        ]
    def __inject__(self) -> str:
        c_code = ["({})".format(child.__inject__()) for child in self.children]
        oprep = Operators.OpMapping[self.op]
        if oprep[1] == 1:
            assert len(c_code) == 1, "Unary operator must have one argument"
            return f"{oprep[0]} {c_code[0]}"
        if oprep[1] == 2:
            assert len(c_code) == 2, "Binary operator must have two arguments"
            return f"{c_code[0]} {oprep[0]} {c_code[1]}"
        if oprep[1] == -1:
            return (f" {oprep[0]} ").join(c_code)
        else:
            _logger.error("Operator arity not yet supported")

# ==============================================================================
# Uclid Complex expressions
# ==============================================================================
class UclidBVSignExtend(UclidExpr):
    def __init__(self, bvexpr, newwidth):
        super().__init__()
        self.bvexpr = bvexpr
        self.newwidth = newwidth
    def __inject__(self) -> str:
        return "bv_sign_extend({}, {})".format(self.newwidth, self.bvexpr.__inject__())
class UclidBVExtract(UclidExpr):
    def __init__(self, bvexpr: UclidExpr, high: int, low: int):
        """Extract a sub-bitvector from a bitvector

        Args:
            bvexpr (UclidExpr): Bitvector expression to extract from
            high (int): high index (inclusive)
            low (int): low index (inclusive)
        """
        super().__init__()
        self.bvexpr = bvexpr
        self.high = high
        self.low = low
    def __inject__(self) -> str:
        return "({})[{}:{}]".format(self.bvexpr.__inject__(), self.high, self.low)

class UclidFunctionApplication(UclidExpr):
    def __init__(self, function, arglist):
        super().__init__()
        self.function = function if isinstance(function, str) else function.name
        self.arglist = arglist
    def __inject__(self) -> str:
        return "{}({})".format(
            self.function, 
            ', '.join([arg.__inject__() for arg in self.arglist])
        )

class UclidArraySelect(UclidExpr):
    def __init__(self, arrayexpr: UclidExpr, indexseq: List[UclidExpr]):
        """Select an element from an array

        Args:
            arrayexpr (UclidExpr): Array-typed expression to select from
            indexseq (List[UclidExpr]): List of indices
        """
        super().__init__()
        self.arrayexpr = arrayexpr if isinstance(arrayexpr, str) else arrayexpr.__inject__()
        self.indexseq = [
            ind if isinstance(ind, UclidExpr) else UclidLiteral(str(ind)) 
            for ind in indexseq
        ]
    def __inject__(self) -> str:
        return "{}[{}]".format(self.arrayexpr, 
            "][".join([ind.__inject__() for ind in self.indexseq])
        )

class UclidArrayUpdate(UclidExpr):
    def __init__(self, arrayexpr: UclidExpr, index: UclidExpr, value: UclidExpr):
        """Update an element in an array

        Args:
            arrayexpr (UclidExpr): Array-typed expression to update
            index (UclidExpr): Index to update (only supported for 1-D arrays)
            value (UclidExpr): New value
        """
        super().__init__()
        self.arrayexpr = arrayexpr if isinstance(arrayexpr, str) else arrayexpr.__inject__()
        self.index = index if isinstance(index, UclidExpr) else UclidLiteral(str(index))
        self.value = value if isinstance(value, UclidExpr) else UclidLiteral(str(value))
    def __inject__(self) -> str:
        return "{}[{} -> {}]".format(self.arrayexpr, self.index.__inject__(), self.value.__inject__())

class UclidRecordSelect(UclidExpr):
    def __init__(self, recexpr: UclidExpr, elemname: str):
        """Select a record element

        Args:
            recexpr (UclidExpr): Record-typed to select from
            elemname (str): Field in the record to select
        """
        super().__init__()
        self.recexpr = recexpr if isinstance(recexpr, str) else recexpr.__inject__()
        self.elemname = elemname
    def __inject__(self) -> str:
        return "{}.{}".format(self.recexpr, self.elemname)
class UclidRecordUpdate(UclidExpr):
    def __init__(self, recexpr: UclidExpr, elemname: str, value: UclidExpr):
        """Update a record element

        Args:
            recexpr (UclidExpr): Record to update
            elemname (str): Field in the record to update
            value (UclidExpr): New value
        """
        super().__init__()
        self.recexpr = recexpr if isinstance(recexpr, str) else recexpr.__inject__()
        self.elemname = elemname
        self.value = value
    def __inject__(self) -> str:
        return "{}[{} := {}]".format(self.recexpr, self.elemname, self.value.__inject__())

class UclidForallExpr(UclidExpr):
    def __init__(self, iterator, typ, bodyexpr : UclidExpr):
        super().__init__()
        self.iterator = iterator if isinstance(iterator, str) else iterator.__inject__()
        self.typ = typ
        self.bodyexpr = bodyexpr
    def __inject__(self) -> str:
        return "forall ({} : {}) :: ({})".format(
            self.iterator, 
            self.typ.__inject__(), 
            self.bodyexpr.__inject__()
        )

# ==============================================================================
# Uclid Literals
# ==============================================================================
class UclidLiteral(UclidExpr):
    """ Uclid literal """
    def __init__(self, lit, isprime = False) -> None:
        super().__init__()
        self.lit = lit if isinstance(lit, str) else str(lit)
        self.isprime = isprime
    def p(self):
        if self.isprime:
            _logger.warn("Double prime for literal {}".format(self.lit))
        return UclidLiteral(self.lit + '\'', True)
    def __inject__(self) -> str:
        return self.lit
    def __add__(self, other):
        return super().__add__(other)

class UclidIntegerLiteral(UclidLiteral):
    """Uclid integer literal"""
    def __init__(self, val):
        super().__init__(val)
        self.val = val
class UclidBooleanLiteral(UclidLiteral):
    """Uclid boolean literal"""
    def __init__(self, val : bool):
        super().__init__("true" if val else "false")
        self.val = val
class UclidBVLiteral(UclidLiteral):
    """Uclid bitvector literal"""
    def __init__(self, val, width: int):
        super().__init__(f'{val}bv{str(width)}')
        self.val = val
        self.width = width

# ==============================================================================
# Uclid Variables
# ==============================================================================
class UclidVar(UclidLiteral):
    def __init__(self, name, typ):
        super().__init__(name)
        self.name = name
        self.typ = typ
    def __inject__(self) -> str:
        return self.name
    def __add__(self, other):
        return super().__add__(other)

# Uclid integer type declaration
class UclidIntegerVar(UclidVar):
    def __init__(self, name) -> None:
        super().__init__(name, UclidIntegerType())
    def __add__(self, other):
        return super().__add__(other)
# Uclid bitvector type declaration
class UclidBVVar(UclidVar):
    def __init__(self, name, width) -> None:
        super().__init__(name, UclidBVType(width))
        self.width = width
    def __add__(self, other):
        return super().__add__(other)

class UclidBooleanVar(UclidVar):
    def __init__(self, name) -> None:
        super().__init__(name, UclidBooleanType())
    def __add__(self, _):
        _logger.error("Addition not supported on Boolean type")
        exit(1)

# Uclid Const declaration
class UclidConst(UclidLiteral):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
    def __inject__(self) -> str:
        return self.name

# ==============================================================================
# Uclid instance variable access
# ==============================================================================
class UclidInstanceVarAccess(UclidExpr):
    def __init__(self, instance : UclidInstance, var: UclidLiteral):
        """Access to a variable in a module instance

        Args:
            instance (UclidInstance/str): Module instance (or its name)
            var (UclidExpr): Variable to access (must be a single variable, not an expression)
        """
        self.instance = instance.__inject__() if isinstance(instance, UclidModule) else instance
        self.var = var
    def __inject__(self) -> str:
        return "{}.{}".format(self.instance, self.var.__inject__())

# ==============================================================================
# Uclid statements
# ==============================================================================
class UclidStmt(UclidElement):
    """ Statements in Uclid. """
    def __init__(self):
        pass
    def __inject__(self) -> str:
        raise NotImplementedError
class UclidComment(UclidStmt):
    def __init__(self, text: str) -> None:
        """Uclid comment

        Args:
            text (str): Comment text
        """
        super().__init__()
        self.text = text
    def __inject__(self) -> str:
        return '\t//'.join(('\n' + self.text.lstrip()).splitlines(True))
class UclidRaw(UclidStmt):
    def __init__(self, rawstring : str):
        """Raw Uclid statement

        Args:
            rawstring (str): Injects this string as is into the Uclid code
        """
        super().__init__()
        self.rawstring = rawstring
    def __inject__(self) -> str:
        return self.rawstring
class UclidEmpty(UclidRaw):
    def __init__(self):
        super().__init__("")
class UclidLocalVarInstanceStmt(UclidStmt):
    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ
    def __inject__(self) -> str:
        return "var {} : {};".format(self.name, self.typ.__inject__())
class UclidAssignStmt(UclidStmt):
    def __init__(self, lval: UclidExpr, rval: UclidExpr) -> None:
        """Assignment statement

        Args:
            lval (UclidExpr): LHS of the assignment
            rval (UclidExpr): RHS of the assignment
        """
        super().__init__()
        self.lval = lval
        if isinstance(rval, UclidExpr):
            self.rval = rval
        elif isinstance(rval, int):
            self.rval = UclidLiteral(str(rval))
        elif isinstance(rval, str):
            self.rval = UclidLiteral(rval)
        else:
            _logger.error(f"Unsupported rval {rval} has type {type(rval)} in UclidAssign")
    def __inject__(self) -> str:
        return "{} = {};".format(self.lval.__inject__(), self.rval.__inject__())

class UclidBlockStmt(UclidStmt):
    def __init__(self, stmts : List[UclidStmt] = []) -> None:
        """Block statement: a sequence of statements

        Args:
            stmts (List[UclidStmt], optional): Statements in this block. Defaults to [].
        """
        super().__init__()
        self.stmts = stmts
    def __inject__(self) -> str:
        return '\n'.join([stmt.__inject__() for stmt in self.stmts])
class UclidCaseStmt(UclidStmt):
    def __init__(self, conditionlist: List[UclidExpr], stmtlist: List[UclidStmt]):
        """Case statement: a sequence of case-split conditions and statements

        Args:
            conditionlist (List[UclidExpr]): List of boolean condition expressions
            stmtlist (List[UclidStmt]): List of statements to execute based on the conditions
        """
        if not isinstance(conditionlist, list) or not isinstance(stmtlist, list):
            _logger.error("UclidCase requires a pair of lists denoting case-split conditions and statements")
            _logger.error("Received {} and {}".format(type(conditionlist), type(stmtlist)))
            exit(1)
        self.conditionlist = conditionlist
        self.stmtlist = stmtlist
    def __inject__(self) -> str:
        cases = [
            "({})\t: {{ \n{} \n}}".format(
                item[0].__inject__(), textwrap.indent(item[1].__inject__(), '\t')
            )
            for item in zip(self.conditionlist, self.stmtlist)
        ]
        return textwrap.dedent(
            """
                case
                {}
                esac
                """
        ).format("\n".join(cases))

class UclidITEStmt(UclidStmt):
    def __init__(self, condition: UclidExpr, tstmt: UclidStmt, estmt: UclidStmt=None):
        """If-then-else statement

        Args:
            condition (UclidExpr): If condition
            tstmt (UclidStmt): Then statement
            estmt (UclidStmt, optional): Else statement. Defaults to None.
        """
        self.condition = condition
        self.tstmt = tstmt
        self.estmt = estmt
    def __inject__(self) -> str:
        if self.estmt == None:
            return """
    if ({}) {{ {} }}
        """.format(
                self.condition.__inject__(), self.tstmt.__inject__()
            )
        else:
            return """
    if ({}) {{ {} }} else {{ {} }}
        """.format(
                self.condition.__inject__(), 
                self.tstmt.__inject__(), 
                self.estmt.__inject__()
            )

class UclidITENestedStmt(UclidStmt):
    def __init__(self, conditionlist: List[UclidExpr], stmtlist: List[UclidStmt]):
        """Nested if-then-else statement (sugaring)

        Args:
            conditionlist (List[UclidExpr]): List of conditions (each nested within the previous else block)
            stmtlist (List[UclidStmt]): List of corresponding statements
        """
        if len(conditionlist) == len(stmtlist):
            self.format = "IT"
        elif len(conditionlist) == len(stmtlist) - 1:
            self.format = "ITE"
        else:
            _logger.error(
                "Illegal lengths of conditionlist and stmt blocks in ITE operator"
            )
        self.conditionlist = conditionlist
        self.stmtlist = stmtlist
    
    def __inject__(self) -> str:
        def ite_rec(clist, slist):
            if len(clist) > 0 and len(slist) > 0:
                nesting = ite_rec(clist[1:], slist[1:])
                return """
    if ({}) {{ {} }} 
    else {{ {} }}""".format(
                    clist[0].__inject__(), slist[0].__inject__(), nesting
                )
            elif len(slist) > 0:
                return "{}".format(slist[0].__inject__())
            elif len(clist) == 0:
                return ""
            else:
                _logger.error("Mismatched clist/slist lengths in ite_rec")
        return ite_rec(self.conditionlist, self.stmtlist)

class UclidProcedureCallStmt(UclidStmt):
    def __init__(self, proc: UclidProcedure, inputs: List[UclidExpr], returns: List[UclidExpr]):
        """Procedure call statement

        Args:
            proc (UclidProcedure): Procedure to call
            inputs (List[UclidExpr]): List of input argument expressions
            returns (List[UclidExpr]): Return variables to assign
        """
        super().__init__()
        self.iname = proc if isinstance(proc, str) else proc.name
        self.inputs = inputs
        self.returns = returns
    def __inject__(self) -> str:
        return "call ({}) = {}({});".format(
            ', '.join([ret.__inject__() for ret in self.returns]),
            self.iname, 
            ', '.join([arg.__inject__() for arg in self.inputs])
        )
class UclidInstanceProcedureCallStmt(UclidStmt):
    def __init__(self, instance: UclidInstance, proc, inputs: List[UclidExpr], returns: List[UclidExpr]):
        """Call procedure from a (sub-)module instance

        Args:
            instance (UclidInstance): Module instance to call
            proc (_type_): Procedure to call from that instance
            inputs (List[UclidExpr]): List of input argument expressions
            returns (List[UclidExpr]): Return variables to assign
        """
        super().__init__()
        self.instance = instance if isinstance(instance, str) else instance.name
        self.iname = "{}.{}".format(
            self.instance, 
            proc if isinstance(proc, str) else proc.name
        )
        self.inputs = inputs
        self.returns = returns
    def __inject__(self) -> str:
        return "call ({}) = {}({});".format(
            ', '.join([ret.__inject__() for ret in self.returns]),
            self.iname, 
            ', '.join([arg.__inject__() for arg in self.inputs])
        )

class UclidNextStmt(UclidStmt):
    def __init__(self, instance: UclidInstance):
        """Next statement: this advances the state of the module instance

        Args:
            instance (UclidInstance): Module instance to advance
        """
        super().__init__()
        self.instance = instance
    def __inject__(self) -> str:
        return "next ({});".format(self.instance.__inject__())
class UclidAssumeStmt(UclidStmt):
    def __init__(self, body: UclidExpr):
        """Assumption statement: this assumes that the body expression is true 
            (constrains executions to only those that satisfy the assumption)

        Args:
            body (UclidExpr): Assumption body
        """
        super().__init__()
        self.body = body
    def __inject__(self) -> str:
        return "assume({});".format(self.body.__inject__())

class UclidAssertStmt(UclidStmt):
    def __init__(self, body: UclidExpr):
        """Assertion statement: this checks that the body expression is true

        Args:
            body (UclidExpr): Assertion body
        """
        super().__init__()
        self.body = body
    def __inject__(self) -> str:
        return "assert({});".format(self.body.__inject__())

class UclidHavocStmt(UclidStmt):
    def __init__(self, variable: UclidVar):
        """Havoc statement: this sets the variable to an arbitrary (non-deterministic) value

        Args:
            variable (UclidVar): Variable to havoc
        """
        super().__init__()
        self.variable = variable
    def __inject__(self) -> str:
        return "havoc {};".format(self.variable.__inject__())

class UclidForStmt(UclidStmt):
    def __init__(self, iterator, iteratortyp, range_low, range_high, body):
        """Uclid for loop statement

        Args:
            iterator (UclidExpr): Iterator for the loop
            iteratortyp (UclidType): Type of the iterator
            range_low (UclidExpr): Lower bound of the loop
            range_high (UclidExpr): Upper bound of the loop
            body (UclidStmt): Body of the loop
        """
        super().__init__()
        self.iterator = iterator
        self.iteratortyp = iteratortyp
        self.range_high = range_high
        self.range_low  = range_low
        self.body = body
    def __inject__(self) -> str:
        return """
for ({} : {}) in range({}, {}) {{
    {}
}}
""".format(
            self.iterator.__inject__(), 
            self.iteratortyp.__inject__(), 
            self.range_low.__inject__(), 
            self.range_high.__inject__(), 
            self.body.__inject__()
        )

# ==============================================================================
# Uclid Control Block
# ==============================================================================
class UclidControlCommand(UclidElement):
    """ Uclid control block commands. """
    def __init__(self):
        pass
    def __inject__(self) -> str:
        raise NotImplementedError
class UclidUnrollCommand(UclidControlCommand):
    def __init__(self, name: str, depth: int):
        """Unroll command

        Args:
            name (str): Name of proof object
            depth (int): Unrolling depth
        """
        _logger.warn("Unroll is deprecated, please use BMC instead")
        self.depth = depth
        self.name = name
    def __inject__(self) -> str:
        return "{} = unroll({});".format(self.name, self.depth)

class UclidBMCCommand(UclidControlCommand):
    def __init__(self, name: str, depth: int):
        """Bounded model checking command

        Args:
            name (str): Name of proof object
            depth (int): Unrolling depth (bound)
        """
        self.depth = depth
        self.name = name
    def __inject__(self) -> str:
        return "{} = bmc({});".format(self.name, self.depth)

class UclidCheckCommand(UclidControlCommand):
    def __init__(self):
        """Uclid check command to execute the proof"""
        super().__init__()
    def __inject__(self) -> str:
        return "check;"

class UclidPrintCexCommand(UclidControlCommand):
    def __init__(self, enginename: str, trace_items : List[UclidExpr] = []):
        """Print counterexample command

        Args:
            enginename (str): Name of proof object
            trace_items (List[UclidExpr], optional): Variables to print from the counterexample trace. Defaults to [] (in which case all module variables are printed).
        """
        super().__init__()
        self.enginename = enginename
        self.trace_items = trace_items
    def __inject__(self) -> str:
        return "{}.print_cex({});".format(
            self.enginename,
            ', '.join([
                item.__inject__() if isinstance(item, UclidExpr) else str(item)
                for item in self.trace_items
            ])
        )

class UclidPrintCexJSONCommand(UclidControlCommand):
    def __init__(self, enginename: str, trace_items : List[UclidExpr] = []):
        """Print counterexample in JSON format

        Args:
            enginename (str): Name of proof object
            trace_items (List[UclidExpr], optional): Variables to print from the counterexample trace. Defaults to [] (in which case all module variables are printed).
        """
        super().__init__()
        self.enginename = enginename
        self.trace_items = trace_items
    def __inject__(self) -> str:
        return "{}.print_cex_json({});".format(
            self.enginename,
            ', '.join([
                item.__inject__() if isinstance(item, UclidExpr) else str(item)
                for item in self.trace_items
            ])
        )
class UclidPrintResultsCommand(UclidControlCommand):
    def __init__(self):
        """Print results command"""
        super().__init__()
    def __inject__(self) -> str:
        return "print_results;"
class UclidControlBlock(UclidElement):
    def __init__(self, stmts : List[UclidControlCommand] = []):
        """Control block: a sequence of control commands

        Args:
            stmts (List[UclidControlCommand], optional): Commands to execute. Defaults to [].
        """
        super().__init__()
        self.stmts = stmts
    def add(self, stmt):
        self.stmts.append(stmt)
    def __inject__(self) -> str:
        return """
control {{
{}
}}""".format(
            textwrap.indent("\n".join([
                stmt.__inject__() for stmt in self.stmts
            ]), '\t')
        )


# ==============================================================================
# LTL Operator Macros
# ==============================================================================
def X(expr):
    return UclidOpExpr("next", [expr])
def F(expr):
    return UclidOpExpr("eventually", [expr])
def G(expr):
    return UclidOpExpr("globally", [expr])

# ==============================================================================
# Uclid Module
# ==============================================================================
class UclidModule(UclidElement):

    def __init__(self, _name, 
        _init = UclidInitBlock([]), 
        _next = UclidNextBlock([]), 
        _control=UclidControlBlock([])):
        """ Uclid module

        Args:
            _name (str): Name of the module
            _init (UclidInitBlock, optional): Init block. Defaults to UclidInitBlock([]).
            _next (UclidNextBlock, optional): next block. Defaults to UclidNextBlock([]).
            _control (UclidControlBlock, optional): control block. Defaults to UclidControlBlock([]).
        """
        super().__init__()
        # Module elements
        self.name       : str               = _name
        self.init       : UclidInitBlock    = _init
        self.next       : UclidNextBlock    = _next
        self.control    : UclidControlBlock = _control
        
        # Module declarations
        # dict: declname : str -> UclidDecl
        self.var_decls      : Dict[str, UclidVarDecl] = dict()
        # dict: declname : str -> UclidDecl
        self.const_decls    : Dict[str, UclidConstDecl] = dict()
        # dict: declname : str -> UclidDecl
        self.type_decls     : Dict[str, UclidTypeDecl] = dict()
        # dict: declname : str -> UclidDecl
        self.instance_decls : Dict[str, UclidInstanceDecl] = dict()
        # dict
        self.ip_var_decls   : Dict[str, UclidVarDecl] = dict() 
        # dict
        self.op_var_decls   : Dict[str, UclidVarDecl] = dict() 
        # list
        self.define_decls   : Dict[str, UclidDefineDecl] = dict()
        # list
        self.function_decls : Dict[str, UclidFunctionDecl] = dict()
        # list
        self.import_decls   : Dict[str, UclidImportDecl] = dict()
        # list
        self.procedure_defns : Dict[str, UclidProcedureDecl] = dict()
        # list
        self.module_assumes : Dict[str, UclidAxiomDecl] = dict()
        # list
        self.module_properties : Dict[str, UclidSpecDecl] = dict()
        
        UclidContext.modules[self.name] = self

    def setInit(self, init: UclidInitBlock) -> None:
        """Set the init block of the module"""
        self.init = init
    
    def setNext(self, next: UclidNextBlock) -> None:
        """Set the next block of the module"""
        self.next = next

    def setControl(self, control: UclidControlBlock) -> None:
        """Set the control block of the module"""
        self.control = control

    def mkUninterpretedType(self, name) -> UclidUninterpretedType:
        """Add a new uninterpreted type to the module

        Args:
            name (str): name of the type

        Returns:
            UclidUninterpretedType: type object
        """
        if name in self.type_decls:
            _logger.warn("Redeclaration of type named {} in module {}".format(name, self.name))
        else:
            t = UclidUninterpretedType(name)
            decl = UclidTypeDecl(name)
            self.type_decls[name] = decl
            return t
    def mkArrayType(self, *args) -> UclidType:
        """Add a new array type to the module

        Args:
            *args: 
                - 2: itype (UclidType): index type, etype (UclidType): element type
                - 3: name (str): type synonym, itype (UclidType): index type, etype (UclidType): element type
        
        Returns:
            UclidArrayType|UclidSynonymType: type object
        """
        if len(args) == 2:
            itype, etype = args
            return UclidArrayType(itype, etype)
        elif len(args) == 3:
            name, itype, etype = args
            if name in self.type_decls:
                _logger.warn("Redeclaration of type named {} in module {}".format(name, self.name))
            else:
                t = UclidArrayType(itype, etype)
                s = UclidSynonymType(name)
                decl = UclidTypeDecl(name, t)
                self.type_decls[name] = decl
                return s
        else:
            _logger.error("Unsupported number of arguments for array type declaration")
    def mkEnumType(self, name: str, members: List[str]) -> UclidEnumType:
        """Add a new enum type to the module

        Args:
            name (str): name of the type
            members (List[str]): list of member names

        Returns:
            UclidEnumType: type object
        """
        if name in self.type_decls:
            _logger.warn("Redeclaration of type named {} in module {}".format(name, self.name))
        else:
            t = UclidEnumType(members)
            s = UclidSynonymType(name)
            decl = UclidTypeDecl(name, t)
            self.type_decls[name] = decl
            return s

    def mkVar(self, name: str, typ: UclidType, porttype=PortType.var) -> UclidVar:
        """Add a new variable to the module

        Args:
            name (str): Variable name
            typ (UclidType): Variable type
            porttype (Portype, optional): var | input | output. Defaults to PortType.var.

        Returns:
            UclidVar: Variable object
        """
        if name in self.var_decls or name in self.op_var_decls or name in self.ip_var_decls:
            _logger.warn("Redeclaration of name {} in module {}".format(name, self.name))
        else:
            v = UclidVar(name, typ)
            decl = UclidVarDecl(name, typ, porttype)
            if porttype == PortType.input:
                self.ip_var_decls[name] = decl
            elif porttype == PortType.output:
                self.op_var_decls[name] = decl
            else:
                self.var_decls[name] = decl
            return v
    def mkIntegerVar(self, name: str, porttype=PortType.var):
        """Add a new integer variable to the module"""
        self.mkVar(name, UclidIntegerType(), porttype)
    def mkBooleanVar(self, name: str, porttype=PortType.var):
        """Add a new boolean variable to the module"""
        self.mkVar(name, UclidBooleanType(), porttype)
    def mkBVVar(self, name: str, width, porttype=PortType.var):
        """Add a new bitvector variable to the module"""
        self.mkVar(name, UclidBVType(width), porttype)

    def mkConst(self, name: str, typ: UclidType, value=None) -> UclidConst:
        """Add a new constant to the module

        Args:
            name (str): Constant name
            typ (UclidType): Constant type
            value (UclidLiteral|str|int, optional): Value of the constant. Defaults to None.

        Returns:
            UclidConst: Constant object
        """
        if name in self.const_decls:
            _logger.warn("Redeclaration of const {} in module {}".format(name, self.name))
        else:
            c = UclidConst(name)
            d = UclidConstDecl(name, typ, value)
            self.const_decls[name] = d
            return c

    def mkInstance(self, name: str, module, argmap: Dict[str, UclidExpr] = {}) -> UclidInstance:
        """Add a new (sub-)module instance to the module

        Args:
            name (str): Instance name
            module (_type_): Module to instantiate
            argmap (Dict[str, UclidExpr], optional): mapping from input/output portnames of submodule to expressions. Defaults to {}.

        Returns:
            UclidInstance: Instance object
        """
        if name in self.instance_decls:
            _logger.warn("Redeclaration of instance {} in module {}".format(name, self.name))
        else:
            decl = UclidInstanceDecl(name, module, argmap)
            inst = UclidInstance(name)
            self.instance_decls[name] = decl
            return inst
    
    def mkImport(self, decltype: DeclTypes, name: str, modulename, refname: str):
        """Add a new import declaration to the module

        Args:
            decltype (DeclTypes): What kind of declaration is this (e.g. type, function, procedure, etc.)?
            name (str): Name of the declaration
            modulename (_type_): Module to import from
            refname (str): Name of import in the source module
        """
        if name in self.import_decls:
            _logger.warn("Redeclaration of import {} in module {}".format(name, self.name))
        else:
            decl = UclidImportDecl(decltype, name, modulename, refname)
            self.import_decls[name] = decl

    def mkDefine(self, name: str, functionsig: UclidFunctionSig, body: UclidStmt) -> UclidDefine:
        """Uclid define declaration

        Args:
            name (str): name of the define macro
            functionsig (UclidFunctionSig): function signature
            body (UclidStmt): body of the define macro

        Returns:
            UclidDefine: define object
        """
        if name in self.define_decls:
            _logger.error("Redeclaration of define macro {} in module {}".format(name, self.name))
        else:
            deffun = UclidDefine(name)
            defdec = UclidDefineDecl(name, functionsig, body)
            self.define_decls.append[name] = defdec
            return deffun
    
    def mkUninterpretedFunction(self, name, functionsig) -> UclidFunction:
        """Uclid uninterpreted function declaration

        Args:
            name (str): name of the function
            functionsig (UclidFunctionSig): function signature

        Returns:
            UclidFunction: function object
        """
        if name in self.function_decls:
            _logger.error("Redeclaration of function {} in module {}".format(name, self.name))
        else:
            uf = UclidFunction(name)
            ufdec = UclidFunctionDecl(name, functionsig)
            self.function_decls[name] = ufdec
            return uf

    def mkProcedure(self, name: str, proceduresig: UclidProcedureSig, body: UclidStmt) -> UclidProcedure:
        """Add a new procedure to the module

        Args:
            name (str): Procedure name
            proceduresig (UclidProcedureSig): Procedure signature
            body (UclidStmt): Procedure body

        Returns:
            UclidProcedure: Procedure object
        """
        if name in self.procedure_defns:
            _logger.error("Redeclaration of procedure {} in module {}".format(name, self.name))
        else:
            proc = UclidProcedure(name)
            procdecl = UclidProcedureDecl(name, proceduresig, body)
            self.procedure_defns[name] = procdecl
            return proc

    def mkAssume(self, name: str, body: UclidExpr) -> UclidAxiomDecl:
        """Add a new assumption to the module

        Args:
            name (str): Name of the assumption (axiom)
            body (UclidExpr): Assumption body
        """
        if name in self.module_assumes:
            _logger.error("Redeclaration of assumption {} in module {}".format(name, self.name))
        else:
            assm = UclidAxiomDecl(name, body)
            self.module_assumes[name] = assm
            return assm

    def mkProperty(self, name, body: UclidExpr, is_ltl=False) -> UclidSpecDecl:
        """Add a new property (assertion) to the module
        
        Args:
            name (str): Name of the specification/assertion
            body (UclidExpr): Specification expression
            is_ltl (bool, optional): Is this an LTL specification? Defaults to False.
        """
        if name in self.module_properties:
            _logger.error("Redeclaration of property {} in module {}".format(name, self.name))
        else:
            spec = UclidSpecDecl(name, body, is_ltl)
            self.module_properties[name] = spec
            return spec

    def __type_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.type_decls.items()])

    def __var_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.var_decls.items()])

    def __const_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.const_decls.items()])

    def __instance_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.instance_decls.items()])

    def __define_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), "\t") 
            for k, decl in self.define_decls.items()])

    def __import_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), "\t") 
            for k, decl in self.import_decls.items()])

    def __procedure_defns__(self):
        return "\n".join([textwrap.indent(defn.__inject__(), "\t")
            for k, defn in self.procedure_defns.items()])

    def __module_assumes__(self):
        return "\n".join([textwrap.indent(assm.__inject__(), "\t") 
            for k, assm in self.module_assumes.items()])

    def __module_properties__(self):
        return "\n".join([textwrap.indent(assr.__inject__(), "\t") 
            for k, assr in self.module_properties.items()])


    def __inject__(self) -> str:
        # dependencies = [inst.obj.module.name for inst in UclidContext.instance_decls[self.name].values()]
        # acc = "\n\n".join([UclidContext.modules[dep].__inject__() for dep in dependencies])
        acc = ""
        init_code = textwrap.indent(self.init.__inject__() if self.init is not None else "", '\t')
        next_code = textwrap.indent(self.next.__inject__() if self.next is not None else "", '\t')
        control_code = textwrap.indent(self.control.__inject__() if self.control is not None else "", '\t')
        decls_code = textwrap.dedent("""
            \t// Imports
            {}

            \t// Types
            {}

            \t// Variables
            {}

            \t// Consts
            {}

            \t// Instances
            {}

            \t// Defines
            {}

            \t// Procedures
            {}

            \t// Assumes
            {}

            \t// Properties
            {}
            """).format(
                self.__import_decls__(),
                self.__type_decls__(),
                self.__var_decls__(),
                self.__const_decls__(),
                self.__instance_decls__(),
                self.__define_decls__(),
                self.__procedure_defns__(),
                self.__module_assumes__(),
                self.__module_properties__()
            )
        acc += textwrap.dedent("""
            module {} {{
            {}

            {}

            {}
                               
            {}
            }}
            """).format(self.name, decls_code, init_code, next_code, control_code)
        return acc
