import json
from unstructured.partition.xlsx import partition_xlsx
import collections
from typing import IO, Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from typing import Any, List
import math
from datetime import datetime
from langchain.document_loaders import UnstructuredExcelLoader
import pandas as pd
from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    Table,
    process_metadata,
)
from unstructured.file_utils.filetype import FileType, add_metadata_with_filetype
from unstructured.partition.common import (
    get_last_modified_date
)

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)

def dict_to_html_table(data):
    table_html = '<table><tr>'
    
    for key,value in data.items():
        table_html += f'<th>{key}</th>'

    table_html += '</tr>\n<tr>'

    for key,value in data.items():
        table_html += f'<td>{value}</td>'

    table_html += '</tr>\n</table>'

    return table_html

def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class ExcelLoader(UnstructuredFileLoader):
    def __init__(
        self, file_path: str, mode='elements', **unstructured_kwargs: Any
    ):
        self.file_path = file_path
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)


    def _get_elements(self) -> List[Element]:
        last_modification_date = None
        sheets = pd.read_excel(self.file_path, sheet_name=None)
        last_modification_date = get_last_modified_date(self.file_path)

        elements: List[Element] = []

        page_number = 0
        for sheet_name, df in sheets.items():
            page_number += 1

            for index, row in df.iterrows():
                # Create a dictionary of the row data
                row_dict = row.to_dict()
                row_cleaned = row_dict.copy()

                for key, value in row_dict.items():
                    if ( 
                        isinstance(value, (int, float)) and math.isnan(value) or
                        (isinstance(value, str) and not value) or
                        value is None
                    ):
                        del row_cleaned[key]

                section = row[0] if len(row) > 0 else None

                html_text = dict_to_html_table(row_cleaned)
                text = json.dumps(row_cleaned, default=serialize_datetime);
                metadata = ElementMetadata(
                    #text_as_html=html_text,
                    section=section,
                    page_name=sheet_name,
                    page_number=page_number,
                    filename=self.file_path,
                    last_modified=last_modification_date,
                )

                table = Table(text=text, metadata=metadata)
                elements.append(table)

        return elements