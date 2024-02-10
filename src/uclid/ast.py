import logging
import textwrap
from enum import Enum

__author__ = "Adwait Godbole"
__copyright__ = "Adwait Godbole"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


class UclidContext:
    # dict:
    #   modulename : str -> declname : str -> UclidDecl
    var_decls = {}
    # dict:
    #   modulename : str -> declname : str -> UclidDecl
    const_decls = {}
    # dict:
    #   modulename : str -> declname : str -> UclidDecl
    typ_decls = {}
    # dict:
    #   modulename : str -> declname : str -> UclidDecl
    instance_decls = {}
    # dict:
    #   modulename : str -> [UclidDecl]list
    ip_var_decls = {}
    # dict:
    #   modulename : str -> [UclidDecl]list
    op_var_decls = {}
    # dict:
    #   modulename : str -> [UclidImport]list
    import_decls = {}
    # dict:
    #   modulename : str -> [UclidDefine]list
    define_decls = {}
    # dict:
    #   modulename : str -> [UclidProcedure]list
    procedure_defns = {}
    # dict:
    #   modulename : str -> [UclidAssume]list
    module_assumes = {}
    # dict:
    #   modulename : str -> UclidModule
    modules = {}
    # name of module in current context
    module_name = "main"

    @staticmethod
    def clearAll():
        """
        Deletes all context information (restart).
        """
        UclidContext.var_decls = {}
        UclidContext.const_decls = {}
        UclidContext.typ_decls = {}
        UclidContext.instance_decls = {}
        UclidContext.ip_var_decls = {}
        UclidContext.op_var_decls = {}
        UclidContext.import_decls = {}
        UclidContext.define_decls = {}
        UclidContext.procedure_defns = {}
        UclidContext.module_assumes = {}
        UclidContext.modules = {}
        UclidContext.module_name = "main"

    @staticmethod
    def setContext(module_name):
        UclidContext.module_name = module_name
        if module_name not in UclidContext.modules:
            UclidContext.var_decls[module_name] = dict()
            UclidContext.const_decls[module_name] = dict()
            UclidContext.typ_decls[module_name] = dict()
            UclidContext.instance_decls[module_name] = dict()
            UclidContext.ip_var_decls[module_name] = []
            UclidContext.op_var_decls[module_name] = []
            UclidContext.import_decls[module_name] = []
            UclidContext.define_decls[module_name] = []
            UclidContext.procedure_defns[module_name] = []
            UclidContext.module_assumes[module_name] = []
            UclidContext.modules[module_name] = None

    @staticmethod
    def __add_typdecl__(name, decl):
        if name in UclidContext.typ_decls[UclidContext.module_name]:
            _logger.warn("Redeclaration of type named {}".format(name))
        else:
            UclidContext.typ_decls[UclidContext.module_name][name] = decl

    @staticmethod
    def __add_vardecl__(name, decl):
        if name in UclidContext.var_decls[UclidContext.module_name]:
            _logger.warn("Redeclaration of name {}".format(name))
        else:
            if decl.porttype == PortType.input:
                _ = UclidContext.ip_var_decls[UclidContext.module_name].append(decl)
            elif decl.porttype == PortType.output:
                _ = UclidContext.op_var_decls[UclidContext.module_name].append(decl)
            UclidContext.var_decls[UclidContext.module_name][name] = decl

    @staticmethod
    def __add_constdecl__(name, decl):
        if name in UclidContext.const_decls[UclidContext.module_name]:
            _logger.warn("Redeclaration of const {}".format(name))
        else:
            UclidContext.const_decls[UclidContext.module_name][name] = decl

    @staticmethod
    def __add_instancedecl__(name, decl):
        if name in UclidContext.instance_decls[UclidContext.module_name]:
            _logger.warn("Redeclaration of instance {}".format(name))
        else:
            UclidContext.instance_decls[UclidContext.module_name][name] = decl

    @staticmethod
    def __add_importdecl__(decl):
        UclidContext.import_decls[UclidContext.module_name].append(decl)

    @staticmethod
    def __add_definedecl__(decl):
        UclidContext.define_decls[UclidContext.module_name].append(decl)

    @staticmethod
    def __add_proceduredefn__(defn):
        UclidContext.procedure_defns[UclidContext.module_name].append(defn)

    @staticmethod
    def __add_moduleassumes__(assm):
        UclidContext.module_assumes[UclidContext.module_name].append(assm)

    # @staticmethod
    # def __add_moduleproperty__(prop):
    #     UclidContext.module_properties[UclidContext.module_name].append(prop)

    @staticmethod
    def __typdecls__(module):
        if module.name not in UclidContext.typ_decls:
            return ""
        return "\n".join(
            [
                textwrap.indent(decl.__inject__(), "\t")
                for k, decl in UclidContext.typ_decls[module.name].items()
            ]
        )

    @staticmethod
    def __vardecls__(module):
        if module.name not in UclidContext.var_decls:
            return ""
        return "\n".join(
            [
                textwrap.indent(decl.__inject__(), "\t")
                for k, decl in UclidContext.var_decls[module.name].items()
            ]
        )

    @staticmethod
    def __constdecls__(module):
        if (
            module.name not in UclidContext.const_decls
            or not UclidContext.const_decls[module.name]
        ):
            return ""
        return "\n".join(
            [
                textwrap.indent(decl.__inject__(), "\t")
                for k, decl in UclidContext.const_decls[module.name].items()
            ]
        )

    @staticmethod
    def __instancedecls__(module):
        if module.name not in UclidContext.instance_decls:
            return ""
        return "\n".join(
            [
                textwrap.indent(decl.__inject__(), "\t")
                for k, decl in UclidContext.instance_decls[module.name].items()
            ]
        )

    @staticmethod
    def __importdecls__(module):
        if module.name not in UclidContext.import_decls:
            return ""
        return "\n".join(
            [
                textwrap.indent(decl.__inject__(), "\t")
                for decl in UclidContext.import_decls[module.name]
            ]
        )

    @staticmethod
    def __definedecls__(module):
        if module.name not in UclidContext.define_decls:
            return ""
        return "\n".join(
            [decl.__inject__() for decl in UclidContext.define_decls[module.name]]
        )

    @staticmethod
    def __proceduredefns__(module):
        if module.name not in UclidContext.procedure_defns:
            return ""
        return "\n".join(
            [
                textwrap.indent(defn.__inject__(), "\t")
                for defn in UclidContext.procedure_defns[module.name]
            ]
        )

    @staticmethod
    def __moduleassumes__(module):
        if module.name not in UclidContext.module_assumes:
            return ""
        return "\n".join(
            [
                textwrap.indent(assm.__inject__(), "\t")
                for assm in UclidContext.module_assumes[module.name]
            ]
        )

    # @staticmethod
    # def __moduleproperties__(module):
    #     if module.name not in UclidContext.module_properties:
    #         return ""
    #     return "\n".join([prop.__inject__() for
    #         prop in UclidContext.module_properties[module.name]])

    @staticmethod
    def __inject__():
        acc = ""
        for modulename in UclidContext.modules:
            acc += UclidContext.modules[modulename].__inject__()
        return acc


setContext = UclidContext.setContext
clearAll = UclidContext.clearAll


class UclidElement:
    def __init__(self) -> None:
        pass

    def __inject__(self):
        raise NotImplementedError


class UclidModule(UclidElement):

    def __init__(self, name, init=None, nextb=None, control=None, properties=[]):
        super().__init__()
        self.name = name
        self.init = init
        self.nextb = nextb
        self.control = control
        self.properties = properties
        UclidContext.modules[self.name] = self

    def __inject__(self):
        # dependencies:
        #   i.obj.module.name for i in UclidContext.instance_decls[self.name].values()
        # acc:
        #   "\n\n".join([UclidContext.modules[dep].__inject__() for dep in dependencies])
        acc = ""
        init_code = textwrap.indent(
            self.init.__inject__() if self.init is not None else "", "\t"
        )
        next_code = textwrap.indent(
            self.nextb.__inject__() if self.nextb is not None else "", "\t"
        )
        control_code = textwrap.indent(
            self.control.__inject__() if self.control is not None else "", "\t"
        )
        decls_code = textwrap.dedent(
            """
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
            """
        ).format(
            UclidContext.__importdecls__(self),
            UclidContext.__typdecls__(self),
            UclidContext.__vardecls__(self),
            UclidContext.__constdecls__(self),
            UclidContext.__instancedecls__(self),
            UclidContext.__definedecls__(self),
            UclidContext.__proceduredefns__(self),
            UclidContext.__moduleassumes__(self),
        )
        prop_code = "\n".join(
            [textwrap.indent(prop.__inject__(), "\t") for prop in self.properties]
        )
        acc += textwrap.dedent(
            """
            module {} {{
            {}

            {}

            {}

            {}

            {}
            }}
            """
        ).format(self.name, decls_code, init_code, next_code, prop_code, control_code)
        return acc


class UclidInit(UclidElement):
    def __init__(self, block) -> None:
        super().__init__()
        if isinstance(block, UclidBlock):
            self.block = block
        elif isinstance(block, list):
            self.block = UclidBlock(block, 1)
        else:
            _logger.error(
                "Unsupported type {} of block in UclidInit".format(type(block))
            )

    def __inject__(self):
        return """
init {{
{}
}}""".format(
            textwrap.indent(self.block.__inject__(), "\t")
        )


class UclidNext(UclidElement):
    def __init__(self, block) -> None:
        super().__init__()
        if isinstance(block, UclidBlock):
            self.block = block
        elif isinstance(block, list):
            self.block = UclidBlock(block, 1)
        elif isinstance(block, UclidStmt):
            self.block = UclidBlock([block], 1)
        else:
            _logger.error(
                "Unsupported type {} of block in UclidNext".format(type(block))
            )

    def __inject__(self):
        return textwrap.dedent(
            """
            next {{
            {}
            }}"""
        ).format(textwrap.indent(self.block.__inject__().strip(), "\t"))


class UclidProperty(UclidElement):
    def __init__(self, name, body, is_ltl=False) -> None:
        super().__init__()
        self.name = name
        self.body = body
        self.is_ltl = is_ltl

    def __inject__(self):
        if not self.is_ltl:
            return "property {} : {};\n".format(self.name, self.body.__inject__())
        return "property[LTL] {} : {};\n".format(self.name, self.body.__inject__())


class UclidModuleAssume(UclidElement):
    def __init__(self, body) -> None:
        super().__init__()
        self.body = body
        UclidContext.__add_moduleassumes__(self)

    def __inject__(self):
        return "    assume ({});".format(self.body.__inject__())


class Operators:
    """
    Comprehensive operator list
    """

    OpMapping = {
        "add": ("+", 2),
        "sub": ("-", 2),
        "umin": ("-", 1),
        "gt": (">", 2),
        "gte": (">=", 2),
        "lt": ("<", 2),
        "lte": ("<=", 2),
        "eq": ("==", 2),
        "neq": ("!=", 2),
        "not": ("!", 1),
        "xor": ("^", 2),
        "and": ("&&", -1),
        "or": ("||", -1),
        "implies": ("==>", 2),
        "bvadd": ("+", 2),
        "bvsub": ("-", 2),
        "bvand": ("&", 2),
        "bvor": ("|", 2),
        "bvnot": ("~", 1),
        "bvlt": ("<", 2),
        "bvult": ("<_u", 2),
        "bvgt": (">", 2),
        "bvugt": (">_u", 2),
        "next": ("X", 1),
        "eventually": ("F", 1),
        "always": ("G", 1),
        "concat": ("++", -1),
    }

    def __init__(self, op_) -> None:
        self.op = op_

    def __inject__(self):
        return Operators.OpMapping[self.op][0]


# Base class for Uclid expressions
class UclidExpr(UclidElement):
    def __inject__(self):
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr("eq", [self, other])
        elif isinstance(other, int):
            return UclidOpExpr("eq", [self, UclidLiteral(str(other))])
        elif isinstance(other, str):
            return UclidOpExpr("eq", [self, UclidLiteral(other)])
        else:
            _logger.error(
                "Unsupported types for operation {}: {} and {}".format(
                    " == ", "UclidExpr", type(other)
                )
            )

    def __ne__(self, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr("neq", [self, other])
        elif isinstance(other, int):
            return UclidOpExpr("neq", [self, UclidLiteral(str(other))])
        elif isinstance(other, str):
            return UclidOpExpr("neq", [self, UclidLiteral(other)])
        else:
            _logger.error(
                "Unsupported types for operation {}: {} ({}) and {}".format(
                    "!=", "UclidExpr", self.__inject__(), type(other)
                )
            )

    def __add__(self, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr("add", [self, other])
        elif isinstance(other, int):
            return UclidOpExpr("add", [self, UclidLiteral(str(other))])
        else:
            _logger.error(
                "Unsupported types for operation {}: {} and {}".format(
                    "+", "UclidExpr", type(other)
                )
            )

    def __invert__(self):
        return UclidOpExpr("not", [self])

    def __and__(self, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr("and", [self, other])
        else:
            _logger.error(
                "Unsupported types for operation {}: {} and {}".format(
                    "&", "UclidExpr", type(other)
                )
            )


# All kinds of operators Operators
class UclidOpExpr(UclidExpr):
    def __init__(self, op, children) -> None:
        super().__init__()
        self.op = op
        self.children = [
            UclidLiteral(str(child)) if isinstance(child, int) else child
            for child in children
        ]

    def __inject__(self):
        children_code = ["({})".format(child.__inject__()) for child in self.children]
        oprep = Operators.OpMapping[self.op]
        if oprep[1] == 1:
            assert len(children_code) == 1, "Unary operator must have one argument"
            return "{} {}".format(oprep[0], children_code[0])
        if oprep[1] == 2:
            assert len(children_code) == 2, "Unary operator must have two arguments"
            return "{} {} {}".format(children_code[0], oprep[0], children_code[1])
        if oprep[1] == -1:
            return (" " + oprep[0] + " ").join(children_code)
        else:
            _logger.error("Operator arity not yet supported")


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


class UclidBVSignExtend(UclidExpr):
    def __init__(self, var, ewidth):
        super().__init__()
        self.var = var
        self.ewidth = ewidth

    def __inject__(self):
        return "bv_sign_extend({}, {})".format(self.ewidth, self.var.__inject__())


class UclidFunctionApply(UclidExpr):
    def __init__(self, func, arglist):
        super().__init__()
        self.iname = func if isinstance(func, str) else func.name
        self.arglist = arglist

    def __inject__(self):
        return "{}({})".format(
            self.iname, ", ".join([arg.__inject__() for arg in self.arglist])
        )


class UclidArraySelect(UclidExpr):
    def __init__(self, array, indexseq):
        super().__init__()
        self.iname = array if isinstance(array, str) else array.__inject__()
        self.indexseq = [
            ind if isinstance(ind, UclidExpr) else UclidLiteral(str(ind))
            for ind in indexseq
        ]

    def __inject__(self):
        return "{}[{}]".format(
            self.iname, "][".join([ind.__inject__() for ind in self.indexseq])
        )


class UclidArrayUpdate(UclidExpr):
    def __init__(self, array, index, value):
        super().__init__()
        self.iname = array if isinstance(array, str) else array.__inject__()
        self.index = index if isinstance(index, UclidExpr) else UclidLiteral(str(index))
        self.value = value if isinstance(value, UclidExpr) else UclidLiteral(str(value))

    def __inject__(self):
        return "{}[{} -> {}]".format(
            self.iname, self.index.__inject__(), self.value.__inject__()
        )


class UclidBVExtract(UclidExpr):
    def __init__(self, bv: UclidExpr, high, low):
        super().__init__()
        self.bv = bv
        self.high = high
        self.low = low

    def __inject__(self):
        return "({})[{}:{}]".format(self.bv.__inject__(), self.high, self.low)


class UclidRecordSelect(UclidExpr):
    def __init__(self, recvar, elemname):
        super().__init__()
        self.recvar = recvar
        self.elemname = elemname

    def __inject__(self):
        return "{}.{}".format(self.recvar.__inject__(), self.elemname)


class UclidForall(UclidExpr):
    def __init__(self, variable, typ, expr: UclidExpr):
        super().__init__()
        self.variable = variable
        self.typ = typ
        self.expr = expr

    def __inject__(self):
        return "forall ({} : {}) :: ({})".format(
            self.variable, self.typ.__inject__(), self.expr.__inject__()
        )


class DeclTypes(Enum):
    VAR = 0
    FUNCTION = 1
    TYPE = 2
    INSTANCE = 3
    SYNTHFUN = 4
    DEFINE = 5
    CONST = 6

    def __inject__(self):
        if self.value == 0:
            return "var"
        elif self.value == 1:
            return "function"
        elif self.value == 2:
            return "type"
        elif self.value == 3:
            return "instance"
        elif self.value == 4:
            return "synthesis function"
        elif self.value == 5:
            return "define"
        elif self.value == 6:
            return "const"
        else:
            _logger.error("Unsupported decl type {}".format(self.name))
            return ""


# Base class for (all sorts of) uclid declarations
class UclidDecl(UclidElement):
    def __init__(self, obj, decltype) -> None:
        super().__init__()
        # name is just a string
        self.name = obj.name
        self.obj = obj
        # body should support the __inject__() operator
        self.declstring = obj.declstring
        self.decltype = decltype
        if self.decltype == DeclTypes.VAR:
            self.porttype = obj.porttype
            self.inject_str = "{} {} : {};".format(
                self.porttype.name, self.name, self.declstring
            )
            UclidContext.__add_vardecl__(self.name, self)
        elif self.decltype == DeclTypes.TYPE:
            self.inject_str = "type {} = {};".format(self.name, self.declstring)
            UclidContext.__add_typdecl__(self.name, self)
        elif self.decltype == DeclTypes.INSTANCE:
            self.inject_str = "instance {} : {};".format(self.name, self.declstring)
            UclidContext.__add_instancedecl__(self.name, self)
        elif self.decltype == DeclTypes.CONST:
            self.inject_str = "const {} : {};".format(self.name, self.declstring)
            UclidContext.__add_constdecl__(self.name, self)
        # TODO: add support for other types
        # TODO: add full type system
        else:
            _logger.error("Currently only var declaration is permitted")
            exit(1)

    def __inject__(self):
        return self.inject_str


class UclidTypeDecl(UclidDecl):
    def __init__(self, obj) -> None:
        super().__init__(obj, DeclTypes.TYPE)


class UclidType:
    def __init__(self, name):
        self.name = name

    def __inject__(self):
        return self.name


class UclidBooleanType(UclidType):
    def __init__(self, name=""):
        super().__init__(name)
        self.declstring = "boolean"
        if name != "":
            self.name = name
            self.decl = UclidTypeDecl(self)
        else:
            self.name = self.declstring


UBool = UclidBooleanType


class UclidIntegerType(UclidType):
    def __init__(self, name=""):
        super().__init__(name)
        self.declstring = "integer"
        if name != "":
            self.name = name
            self.decl = UclidTypeDecl(self)
        else:
            self.name = self.declstring


class UclidBVType(UclidType):
    def __init__(self, width, name=""):
        super().__init__(name)
        self.width = width
        self.declstring = "bv{}".format(self.width)
        if name != "":
            self.name = name
            self.decl = UclidTypeDecl(self)
        else:
            self.name = self.declstring


Ubv = UclidBVType


class UclidArrayType(UclidType):
    def __init__(self, itype, etype, name=""):
        super().__init__(name)
        self.indextype = itype
        self.elemtype = etype
        self.declstring = "[{}]{}".format(
            self.indextype.__inject__(), self.elemtype.__inject__()
        )
        if name != "":
            self.name = name
            self.decl = UclidTypeDecl(self)
        else:
            self.name = self.declstring


class UclidEnumType(UclidType):
    def __init__(self, members, name=""):
        super().__init__(name)
        self.members = members
        if name != "":
            self.name = name
            self.declstring = "enum {{ {} }}".format(", ".join(self.members))
            self.decl = UclidTypeDecl(self)
        else:
            _logger.error("Enum type must be a named type in UclidEnumType!")


class UclidLiteralType(UclidType):
    def __init__(self, name):
        super().__init__(name)
        self.declstring = name


class UclidFunctionType(UclidType):
    """
    Function signature type
    """

    def __init__(self, ip_args, out_type) -> None:
        # Function type does not have a declaration
        #   so it also does not require a name/declstring
        super().__init__("")
        self.ip_args = ip_args
        self.out_type = out_type

    def __inject__(self):
        input_sig = ", ".join(
            ["{} : {}".format(i[0], i[1].__inject__()) for i in self.ip_args]
        )
        return "({}) : {}".format(input_sig, self.out_type.__inject__())


class UclidProcedureType(UclidType):
    # inputs_args are pairs of str, Uclidtype elements
    def __init__(self, ip_args, modify_vars, return_vars):
        # Procedure type does not have a declaration
        #   so it also does not require a name/declstring
        super().__init__("")
        self.ip_args = ip_args
        self.modify_vars = modify_vars
        self.return_vars = return_vars

    def __inject__(self):
        input_sig = ", ".join(
            [
                "{} : {}".format(
                    i[0] if isinstance(i[0], str) else i[0].lit, i[1].declstring
                )
                for i in self.ip_args
            ]
        )
        modify_sig = (
            "\nmodifies {};".format(
                ", ".join([sig.__inject__() for sig in self.modify_vars])
            )
            if len(self.modify_vars) != 0
            else ""
        )
        return_sig = (
            "\nreturns ({})".format(
                ", ".join(
                    ["{} : {}".format(i[0], i[1].declstring) for i in self.return_vars]
                )
            )
            if len(self.return_vars) != 0
            else ""
        )
        return "({}){}{}".format(input_sig, modify_sig, return_sig)


class PortType(Enum):
    var = 0
    input = 1
    output = 2

    def __inject__(self):
        if self.value == PortType.var:
            return "var"
        elif self.value == PortType.input:
            return "input"
        elif self.value == PortType.output:
            return "output"
        else:
            _logger.error("Unsupported port type")


class UclidInstance(UclidElement):
    """
    Module instances in Uclid
    """

    def __init__(self, name, module, argmap) -> None:
        super().__init__()
        self.name = name
        self.module = module
        self.argmap = argmap
        modname = module.name
        # print(argmap)
        # print([i.__inject__() for i in UclidContext.ip_var_decls[modname]])
        self.declstring = "{}({})".format(
            modname,
            ", ".join(
                [
                    "{} : ({})".format(port.name, argmap[port.name].__inject__())
                    for port in UclidContext.ip_var_decls[modname]
                    + UclidContext.op_var_decls[modname]
                ]
            ),
        )
        self.decl = UclidDecl(self, DeclTypes.INSTANCE)

    def __inject__(self):
        return self.name


class UclidInstanceRaw(UclidElement):
    """
    Raw (external) module instances in Uclid
    """

    def __init__(self, name, module, argmap) -> None:
        super().__init__()
        self.name = name
        self.argmap = argmap
        modname = module.name if isinstance(module, UclidModule) else module
        # print(argmap)
        # print([i.__inject__() for i in UclidContext.ip_var_decls[modname]])
        self.declstring = "{}({})".format(
            modname,
            ", ".join(
                ["{} : ({})".format(portname, argmap[portname]) for portname in argmap]
            ),
        )
        self.decl = UclidDecl(self, DeclTypes.INSTANCE)

    def __inject__(self):
        return self.name


class UclidInstanceVarAccess(UclidExpr):
    def __init__(self, instance, var):
        self.instance = (
            instance.__inject__() if isinstance(instance, UclidModule) else instance
        )
        self.var = var

    def __inject__(self):
        return "{}.{}".format(self.instance, self.var.__inject__())


class UclidImport(UclidElement):
    """
    Import declarations from other modules
    """

    def __init__(self, impobj, modulename, refname=None) -> None:
        super().__init__()
        self.impobj = impobj
        self.modulename = modulename
        self.refname = refname if refname is not None else impobj.name
        UclidContext.__add_importdecl__(self)

    def __inject__(self):
        return "{} {} = {}.{};".format(
            self.impobj.decl.decltype.__inject__(),
            self.impobj.name,
            self.modulename,
            self.refname,
        )


class UclidWildcardImport(UclidImport):
    """
    Import declarations from other modules
    """

    def __init__(self, type_: DeclTypes, modulename) -> None:
        self.type = type_
        self.modulename = modulename
        UclidContext.__add_importdecl__(self)

    def __inject__(self):
        return "{} * = {}.*;".format(self.type.__inject__(), self.modulename)


class UclidDefine(UclidElement):
    """
    Define (function) declarations
    """

    def __init__(
        self, name: str, function_sig: UclidFunctionType, body: UclidExpr
    ) -> None:
        super().__init__()
        self.name = name
        self.function_sig = function_sig
        self.body = body
        UclidContext.__add_definedecl__(self)

    def __inject__(self):
        return "\tdefine {}{} = {};".format(
            self.name, self.function_sig.__inject__(), self.body.__inject__()
        )


# A named literal (WYSIWYG)
class UclidLiteral(UclidExpr):
    """
    Literals and variables in Uclid
    """

    def __init__(self, lit, isprime=False) -> None:
        super().__init__()
        self.lit = lit if isinstance(lit, str) else str(lit)
        self.isprime = isprime

    def p(self):
        if self.isprime:
            _logger.warn("Double prime for literal {}".format(self.lit))
        return UclidLiteral(self.lit + "'", True)

    def __inject__(self):
        return self.lit

    def __add__(self, other):
        return super().__add__(other)


class UclidIntegerConst(UclidLiteral):
    def __init__(self, val):
        super().__init__(val)


class UclidBooleanConst(UclidLiteral):
    def __init__(self, val: bool):
        super().__init__(str(val).lower())


class UclidBVConst(UclidLiteral):
    def __init__(self, val, width: int):
        super().__init__(val)
        self.width = width
        self.lit = f"{self.lit}bv{str(self.width)}"


# Uclid literal which must be declared as a variable
class UclidVar(UclidLiteral):
    def __init__(self, name, typ: UclidType, porttype=PortType.var):
        super().__init__(name)
        self.name = name
        self.typ = typ
        self.porttype = porttype
        self.declstring = typ.name
        _ = UclidDecl(self, DeclTypes.VAR)

    def __inject__(self):
        return self.name

    def __add__(self, other):
        return super().__add__(other)


def mkVar(varname: list, typ, porttype):
    return UclidVar(varname, typ, porttype)


def mkVars(varnames: list, typ, porttype):
    return [UclidVar(varname, typ, porttype) for varname in varnames]


class UclidConst(UclidLiteral):
    def __init__(self, name: str, typ: UclidType, value=None):
        super().__init__(name)
        self.name = name
        self.typ = typ
        if value is None:
            self.declstring = typ.name
        else:
            self.declstring = "{} = {}".format(
                typ.name,
                value.__inject__() if isinstance(value, UclidExpr) else str(value),
            )
        _ = UclidDecl(self, DeclTypes.CONST)

    def __inject__(self):
        return self.name


# Uclid integer type declaration
class UclidIntegerVar(UclidVar):
    def __init__(self, name, porttype=PortType.var) -> None:
        super().__init__(name, UclidIntegerType(), porttype)

    def __add__(self, other):
        return super().__add__(other)


def mkIntVar(varname: str, porttype=PortType.var):
    return UclidIntegerVar(varname, porttype)


def mkIntVars(varnames: list, porttype=PortType.var):
    return [UclidIntegerVar(varname, porttype) for varname in varnames]


# Uclid bitvector type declaration
class UclidBVVar(UclidVar):
    def __init__(self, name, width, porttype=PortType.var) -> None:
        super().__init__(name, UclidBVType(width), porttype)
        self.width = width

    def __add__(self, other):
        return super().__add__(other)


class UclidBooleanVar(UclidVar):
    def __init__(self, name, porttype=PortType.var) -> None:
        super().__init__(name, UclidBooleanType(), porttype)

    def __add__(self, _):
        _logger.error("Addition not supported on Boolean type")
        # return super().__add__(other)


class UclidComment(UclidElement):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def __inject__(self):
        return "\t//{}\n".format(self.text)


class UclidStmt(UclidElement):
    """
    Statements in Uclid.
    """

    def __init__(self):
        pass

    def __inject__(self):
        raise NotImplementedError


class UclidRaw(UclidStmt):
    def __init__(self, stmt: str):
        super().__init__()
        self.stmt = stmt

    def __inject__(self):
        return self.stmt


class UclidEmpty(UclidRaw):
    def __init__(self):
        super().__init__("")


class UclidLocalVarInst(UclidStmt):
    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ

    def __inject__(self):
        return "var {} : {};".format(self.name, self.typ.__inject__())


class UclidAssign(UclidStmt):
    def __init__(self, lval, rval) -> None:
        super().__init__()
        self.lval = lval
        if isinstance(rval, UclidExpr):
            self.rval = rval
        elif isinstance(rval, int):
            self.rval = UclidLiteral(str(rval))
        elif isinstance(rval, str):
            self.rval = UclidLiteral(rval)
        else:
            _logger.error("Unsupported rval type {} in UclidAssign".format(type(rval)))

    def __inject__(self):
        return "{} = {};".format(self.lval.__inject__(), self.rval.__inject__())


class UclidBlock(UclidStmt):
    def __init__(self, stmts=[], indent=0) -> None:
        super().__init__()
        self.stmts = stmts
        self.indent = indent

    def __inject__(self):
        return ("\n" + self.indent * "\t").join(
            [stmt.__inject__() for stmt in self.stmts]
        )


class UclidCase(UclidStmt):
    def __init__(self, conditionlist, stmts):
        if not isinstance(conditionlist, list) or not isinstance(stmts, list):
            _logger.error(
                "UclidCase requires a pair of lists denoting case-split conditions/stmts"
            )
            print([cond.__inject__() for cond in conditionlist])
        self.conditionlist = conditionlist
        self.stmts = stmts

    def __inject__(self):
        cases = [
            "({})\t: {{ \n{} \n}}".format(
                item[0].__inject__(), textwrap.indent(item[1].__inject__(), "\t")
            )
            for item in zip(self.conditionlist, self.stmts)
        ]
        return textwrap.dedent(
            """
                case
                {}
                esac
                """
        ).format("\n".join(cases))


class UclidITE(UclidStmt):
    def __init__(self, condition, tstmt, estmt=None):
        self.condition = condition
        self.tstmt = tstmt
        self.estmt = estmt

    def __inject__(self):
        if self.estmt is None:
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
                self.estmt.__inject__(),
            )


class UclidITENested(UclidStmt):
    def __init__(self, conditionlist, stmtlist):
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

    def __inject__(self):
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


class UclidProcedure(UclidElement):
    def __init__(self, name: str, typ: UclidProcedureType, body: UclidStmt):
        super().__init__()
        self.name = name
        self.typ = typ
        self.body = body
        UclidContext.__add_proceduredefn__(self)

    def __inject__(self):
        return """procedure {} {}
{{
{}
}}
    """.format(
            self.name,
            self.typ.__inject__(),
            textwrap.indent(self.body.__inject__(), "\t"),
        )


class UclidProcedureCall(UclidStmt):
    def __init__(self, proc, ip_args, ret_vals):
        super().__init__()
        self.iname = proc if isinstance(proc, str) else proc.name
        self.ip_args = ip_args
        self.ret_vals = ret_vals

    def __inject__(self):
        return "call ({}) = {}({});".format(
            ", ".join([ret.__inject__() for ret in self.ret_vals]),
            self.iname,
            ", ".join([arg.__inject__() for arg in self.ip_args]),
        )


class UclidInstanceProcedureCall(UclidStmt):
    def __init__(self, instance, proc, ip_args, ret_vals):
        super().__init__()
        self.instance = instance if isinstance(instance, str) else instance.name
        self.iname = "{}.{}".format(
            self.instance, proc if isinstance(proc, str) else proc.name
        )
        self.ip_args = ip_args
        self.ret_vals = ret_vals

    def __inject__(self):
        return "call ({}) = {}({});".format(
            ", ".join([ret.__inject__() for ret in self.ret_vals]),
            self.iname,
            ", ".join([arg.__inject__() for arg in self.ip_args]),
        )


class UclidNextStmt(UclidStmt):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def __inject__(self):
        return "next ({});".format(self.instance.__inject__())


class UclidAssume(UclidStmt):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def __inject__(self):
        return "assume({});".format(self.body.__inject__())


class UclidAssert(UclidStmt):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def __inject__(self):
        return "assert({});".format(self.body.__inject__())


class UclidHavoc(UclidStmt):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    def __inject__(self):
        return "havoc {};".format(self.variable.__inject__())


class UclidFor(UclidStmt):
    def __init__(self, range_variable, range_typ, range_low, range_high, body):
        super().__init__()
        self.range_high = range_high
        self.range_low = range_low
        self.range_variable = range_variable
        self.range_typ = range_typ
        self.body = body

    def __inject__(self):
        return """
for ({} : {}) in range({}, {}) {{
    {}
}}
""".format(
            self.range_variable.__inject__(),
            self.range_typ.__inject__(),
            self.range_low.__inject__(),
            self.range_high.__inject__(),
            self.body.__inject__(),
        )


class UclidControlCommand(UclidElement):
    """
    Uclid control block commands.
    """

    def __init__(self):
        pass

    def __inject__(self):
        raise NotImplementedError


class UclidControlBlock(UclidControlCommand):
    def __init__(self, stmts=[]):
        super().__init__()
        self.stmts = stmts

    def add(self, stmt):
        self.stmts.append(stmt)

    def __inject__(self):
        return """
control {{
{}
}}""".format(
            textwrap.indent("\n".join([stmt.__inject__() for stmt in self.stmts]), "\t")
        )


class UclidUnrollCommand(UclidControlCommand):
    def __init__(self, name, depth):
        self.depth = depth
        self.name = name

    def __inject__(self):
        return "{} = unroll({});".format(self.name, self.depth)


class UclidBMCCommand(UclidControlCommand):
    def __init__(self, name, depth):
        self.depth = depth
        self.name = name

    def __inject__(self):
        return "{} = bmc({});".format(self.name, self.depth)


class UclidCheckCommand(UclidControlCommand):
    def __init__(self):
        super().__init__()

    def __inject__(self):
        return "check;"


class UclidPrintCexCommand(UclidControlCommand):
    def __init__(self, engine, trace_items=[]):
        super().__init__()
        self.engine = engine
        self.trace_items = trace_items

    def __inject__(self):
        return "{}.print_cex({});".format(
            self.engine.name,
            ", ".join(
                [
                    item.__inject__() if isinstance(item, UclidExpr) else str(item)
                    for item in self.trace_items
                ]
            ),
        )


class UclidPrintCexJSONCommand(UclidControlCommand):
    def __init__(self, engine, trace_items=[]):
        super().__init__()
        self.engine = engine
        self.trace_items = trace_items

    def __inject__(self):
        return "{}.print_cex_json({});".format(
            self.engine.name,
            ", ".join(
                [
                    item.__inject__() if isinstance(item, UclidExpr) else str(item)
                    for item in self.trace_items
                ]
            ),
        )


class UclidPrintResultsCommand(UclidControlCommand):
    def __init__(self):
        super().__init__()

    def __inject__(self):
        return "print_results;"


# LTL Operator Macros
def X(expr):
    return UclidOpExpr("next", [expr])


def F(expr):
    return UclidOpExpr("eventually", [expr])


def G(expr):
    return UclidOpExpr("globally", [expr])


CMD_check = UclidCheckCommand()
CMD_print = UclidPrintResultsCommand()
