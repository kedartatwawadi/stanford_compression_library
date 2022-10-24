from dataclasses import dataclass
from typing import Any


@dataclass
class BinaryNode:
    left_child: Any = None
    right_child: Any = None
    id: Any = None

    @property
    def is_leaf_node(self):
        return (self.left_child is None) and (self.right_child is None)

    def _get_lines(self):
        """
        internal function to visualize the tree starting from the root node.

        Returns:
            lines, root_node
        """

        # utility function to merge two lists of strings
        def merge_lines(lines_1, lines_2):
            """
            example: lines_1 = ["A","B","C"],
            lines_2 = ["1","2","3"]
            -> return lines = ["A1","B2","C3"]
            """
            lines = []
            for l1, l2 in zip(lines_1, lines_2):
                lines.append(l1 + str(l2))
            return lines

        # form a printable id (in case it is None)
        _id = self.id if self.id else "\u00B7"  # -> center dot

        # if node is leaf, return only the id
        if self.is_leaf_node:
            return [_id], 0

        # recursively get lines, and root node location for the
        # left and right subtrees
        if self.left_child is not None:
            lines_left, root_loc_left = self.left_child._get_lines()
        else:
            lines_left, root_loc_left = [], None

        if self.right_child is not None:
            lines_right, root_loc_right = self.right_child._get_lines()
        else:
            lines_right, root_loc_right = [], None

        ## The strategy to join the two subtrees is:
        #    |--left_tree
        # id-|
        #    |--right_tree
        #
        ## Step 0: Join the lines together (with a space in between)
        #       left_tree
        #
        #       right_tree
        #
        ## Step 1: add the first stage
        #
        #     --left_tree
        #
        #     --right_tree
        #
        ## Step 2: add the second stage
        #
        #    |--left_tree
        #    |
        #    |--right_tree
        #
        ## Step 3: Finally add the id
        #    |--left_tree
        # id-|
        #    |--right_tree
        #

        ## join the two lines
        lines = lines_left + [""] + lines_right
        root_node_loc = len(lines_left)

        # update right loc if it is not None
        if root_loc_right is not None:
            root_loc_right = root_node_loc + 1 + root_loc_right

        # add the first stage
        spacer = ["  " for i in range(len(lines))]
        if root_loc_left is not None:
            spacer[root_loc_left] = "--"
        if root_loc_right is not None:
            spacer[root_loc_right] = "--"
        lines = merge_lines(spacer, lines)

        # add the second stage
        spacer = [" " for i in range(len(lines))]
        if root_loc_left is not None:
            for i in range(root_loc_left, root_node_loc + 1):
                spacer[i] = "|"

        if root_loc_right is not None:
            for i in range(len(lines_left) + 1, root_loc_right + 1):
                spacer[i] = "|"
        lines = merge_lines(spacer, lines)

        # add the final stage
        _id = _id + "-"
        spacer = [" " * len(_id) for i in range(len(lines))]
        spacer[root_node_loc] = _id
        lines = merge_lines(spacer, lines)

        return lines, root_node_loc

    def print_node(self):
        """
        Print the tree from the root node

        returns the lines of the tree, and the line number of the node
        internal function used in recursively printing the node

                    |--D
               |--·-|
               |    |--C
          |--·-|
          |    |--B
        ·-|
          |--A
        """
        lines, _ = self._get_lines()
        print()  # add empty line to make sure we start on newline
        for line in lines:
            print(line)
