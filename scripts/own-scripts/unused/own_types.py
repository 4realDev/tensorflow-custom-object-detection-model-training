# Problem 1: Forward Referencing
from typing import Optional, List, Any


class BinaryTree:
    value: int
    # throws an Runtime Error -> NameError (name 'BinaryTree' is not defined)
    # and a TypeError with mypy because we use BinaryTree inside itself, where it is not fully defined yet
    # solve it by using string with type name 'BinaryTree' in parentheses instead
    left: BinaryTree
    right: BinaryTree

# Problem 2: Using None for values, which shouldn't be None
# Incompatible types in assignment (expression has type "None", variable has type "BinaryTree")
# solve it with python typing module and its Optional DataClass
# (offers different predefined DataClasses to express special case and make typing more conventient)


class BinaryTree:
    value: int
    # throws an error with mypy because value is not allowed to be none
    left: 'BinaryTree' = None
    right: 'BinaryTree' = None


class BinaryTree:
    value: int
    left: 'Optional[BinaryTree]' = None  # some type of none
    right: 'Optional[BinaryTree]' = None


# Union of types
T = Union[int, float]


@dataclass
class Item_Position:
    ymin: int
    ymax: int
    xmin: int
    xmax: int


@dataclass
class Cropped_Img_Data:
    position: Item_Position
    color: str
    name: str
    ocr_recognized_text: str


class Board_Data_Links:
    self: str  # "https://api.miro.com/v2/boards/uXjVPOIamMU=",
    related: str  # "https://api.miro.com/v2/boards/uXjVPOIamMU=/members?limit=20&offset=0"


class Board_Data_User:
    id: str  # "3458764516565255692"
    type: str  # "user"
    name: str  # "Vladimir"


class CurrentUserMembership:
    id: str  # "3458764516565255692"
    type: str  # "board_member"
    name: str  # "Vladimir"
    role: str  # "owner"


class PermissionsPolicy:
    collaborationToolsStartAccess: str  # "all_editors"
    copyAccess: str  # "anyone"
    copyAccessLevel: str  # "anyone"
    sharingAccess: str  # "team_members_with_editing_rights"


class SharingPolicy:
    access: str  # "private"
    inviteToAccountAndBoardLinkAccess: str  # "no_access"
    organizationAccess: str  # "private"
    teamAccess: str  # "private"


class Policy:
    permissionsPolicy: PermissionsPolicy
    sharingPolicy: SharingPolicy


class Team:
    id: str  # "3458764523285765924"
    type: str  # "team"
    name: str  # "Dev team"


class Board_Data:
    id: str  # "uXjVPOIamMU="
    type: str  # "board"
    name: str  # "2022-10-12-15-23-21"
    description: str  # "2022-10-12-15-23-21"
    links: Board_Data_Links
    createdAt: str  # "2022-10-12T13:23:38Z"
    createdBy: Board_Data_User
    currentUserMembership: CurrentUserMembership
    modifiedAt: str  # "2022-10-12T13:26:12Z"
    modifiedBy: Board_Data_User
    owner: Board_Data_User
    permissionsPolicy: PermissionsPolicy
    policy: Policy
    sharingPolicy: SharingPolicy
    team: Team
    viewLink: str  # "https://miro.com/app/board/uXjVPOIamMU="


class Board_Links:
    # "https://api.miro.com/v2/boards?query=&owner=&limit=20&offset=0&sort=default{&team_id}",
    self: str
    # "https://api.miro.com/v2/boards?query=&owner=&limit=20&offset=20&sort=default{&team_id}",
    next: str
    # "https://api.miro.com/v2/boards?query=&owner=&limit=20&offset=20&sort=default{&team_id}"
    last: str


class Get_All_Boards_Response:
    size: int  # 20
    offset: int  # 0
    limit: int  # 20
    total: int  # 26
    data: List[Board_Data]
    links: Board_Links
    type: Any
