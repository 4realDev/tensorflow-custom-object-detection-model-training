# pip install requests

import aiohttp
import config
import nest_asyncio
import asyncio
import requests
import json
from dataclasses import dataclass
from sklearn.feature_extraction import img_to_graph
from aiohttp import FormData

nest_asyncio.apply()


# board_name = "STICKY NOTES SYNC BOARD"
# board_description = "Board to test the synchronization of the sticky notes in the real life and the here created miro board."


# VARS FOR MIRO REST API
DEBUG_PRINT_RESPONSES = False
auth_token = "eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_N9OybOclP4WmwOKCNUjVuVMDshE"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}"
}


# MIRO REST API HELPER FUNCTIONS
# GET ALL BOARD IDS AND NAMES
# WARNING:
# seems that the Get boards REST API function is not working properly
# sometimes it only returns one board, even if there are more


async def get_all_miro_board_names_and_ids(session):
    url = "https://api.miro.com/v2/boards?limit=50&sort=default"
    async with session.get(url, headers=headers) as resp:
        response = await resp.json()

        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return [{
            "name": board['name'],
            "id": board['id']
        } for board in response['data']]


# GET ALL BOARD ITEMS
# default limit is maximum (50)
async def get_all_items(
    item_type: str,
    board_id: str,
    session: aiohttp.client.ClientSession,
    max_num_of_items: int = 50
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D' )}/items?limit={max_num_of_items}&type={item_type}"

    async with session.get(url, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return response['data']


# CREATE FRAME
async def create_frame(
    pos_x: float,
    pos_y: float,
    title: str,
    height: float,
    width: float,
    board_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/frames"
    payload = {
        "data": {
            "format": "custom",
            "title": title,
            "type": "freeform"
        },
        "style": {"fillColor": "#ffffffff"},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "height": height,
            "width": width
        }
    }

    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "authorization": f"Bearer {auth_token}"
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        if resp.status == requests.codes.created:
            print(
                f"Successfully created frame named {title}.")
        return response['id']


# CREATE ITEM
async def create_sticky_note(
    pos_x: float,
    pos_y: float,
    width: float,
    color: str,
    text: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/sticky_notes"
    payload = {
        "data": {
            "content": text,
            "shape": "square"
        },
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "width": width
        },
        "parent": {"id": parent_id}
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created sticky note with with the text {text}.")

        return resp.status


# CREATE LINE
async def create_line(
    pos_x: float,
    pos_y: float,
    width: float,
    height: float,
    color: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/shapes"

    payload = {
        "data": {"shape": "round_rectangle"},
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "height": height,
            "width": width
        },
        "parent": {"id": parent_id}
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        return resp.status


async def create_image(
    pos_x: float,
    pos_y: float,
    width: float,
    title: str,
    path: str,
    board_id: str,
    parent_id: str,
    session: aiohttp.client.ClientSession
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/images"

    headers = {
        "accept": "*/*",
        "authorization": f"Bearer {auth_token}"
    }

    payload = {
        "title": title,
        "position": {
            "x": pos_x,
            "y": pos_y,
            "origin": "center"
        },
        "geometry": {
            "width": width,
            "rotation": 0
        },
        "parent": {"id": parent_id}
    }

    # fmt: off
    data = FormData()
    data.add_field('resource', open(path, "rb"), filename=f'{title}.png', content_type="application/png")
    data.add_field('data', json.dumps(payload), content_type="application/json")
    # fmt: on

    async with session.post(url=url,  data=data, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        # if resp.status == requests.codes.created:
        #     print(
        #         f"Successfully created image of the sticky note with the path {path}.")

        return resp.status


# CREATE NEW MIRO-BOARD
# (if no Board with given name exists, else return the id or the existing one)
async def create_new_miro_board_or_get_existing(
    name: str,
    description: str,
    save_in_existing_miro_board: bool,
    session: aiohttp.client.ClientSession
) -> str:
    board_names_and_ids = await asyncio.create_task(get_all_miro_board_names_and_ids(session))
    print(f"\nExisting Boards: {board_names_and_ids}")

    # 1. save_in_existing_miro_board flag is set manually to "True" (default is "False")
    #    search for the given board name inside all existing miro boards
    if save_in_existing_miro_board:
        # 1.1 Board with the board given name exists -> return its id
        for board_name_and_id in board_names_and_ids:
            if board_name_and_id['name'] == name:
                print(
                    f"\nINFO: The board with the name {name} already exist. \n")
                return board_name_and_id['id']

        # 1.2 Board with the given board name does not exist -> return ERROR and stop function
        print(f"\n ERROR: The 'Save in existing Board' checkbox is set to {save_in_existing_miro_board} and the given miro board name {name} was not found inside all miro boards. It could be possible that the searched board does not exist or must be still indexed from MIRO. Please wait a few seconds and try again or uncheck the 'Save in existing Board' checkbox to create a new miro board with the given name. \n")
        return "-1"

    # 2. save_in_existing_miro_board flag is "False" -> create a new miro board with the given name
    #    (no matter if the board with this name already exist)
    url = "https://api.miro.com/v2/boards"

    payload = {
        "name": name,
        "description": description,
        "policy": {
            "permissionsPolicy": {
                "collaborationToolsStartAccess": "all_editors",
                "copyAccess": "anyone",
                "sharingAccess": "team_members_with_editing_rights"
            },
            "sharingPolicy": {
                "access": "private",
                "inviteToAccountAndBoardLinkAccess": "no_access",
                "organizationAccess": "private",
                "teamAccess": "private"
            }
        }
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

        if resp.status == requests.codes.created:
            print(
                f"Successfully created new miro board with the name {name} and the board_id {response['id']}.")

    return response['id']


# !!! NOT IN USE !!!
# DELETE BOARD ITEM
async def delete_item(item_id: str):
    global global_board_id
    global global_session
    url: str = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/items/{item_id}"

    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_Bw1UPxNElmWbBGy8MfSWWJWOCLs"
    }

    response = requests.delete(url, headers=headers)
    if DEBUG_PRINT_RESPONSES:
        print(await response.text())

    # WARNING: Seems not to work with global_session.delete(url, headers=headers)
    #     async with global_session.delete(url, headers=headers) as resp:
    #         response = await resp.json()
    #         if DEBUG_PRINT_RESPONSES: print(await resp.text())


# DELETE ALL BOARD ITEMS
async def delete_all_items(item_type: str):
    global global_session
    board_items = await asyncio.create_task(get_all_items(item_type))
    for board_item in board_items:
        await asyncio.create_task(delete_item(board_item['id']))


# CREATE LIST OF ITEMS
async def create_all_sticky_notes(sticky_note_positions):
    global global_session
    global global_board_id

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/sticky_notes"

    for sticky_note_position in sticky_note_positions:
        payload = {
            "data": {"shape": "square"},
            "position": {
                "origin": "center",
                "x": sticky_note_position['xmin'],
                "y": sticky_note_position['ymin']
            },
            "geometry": {
                #             "height": sticky_note_position['ymax'] - sticky_note_position['ymin'],
                "width": sticky_note_position['xmax'] - sticky_note_position['xmin']
            }
        }

        async with global_session.post(url, json=payload, headers=headers) as resp:
            response = await resp.json()
            if DEBUG_PRINT_RESPONSES:
                print(await resp.text())
