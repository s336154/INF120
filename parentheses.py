# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:29:54 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com

"""


def check_parentheses(text):
    parentheses = [char for char in text if char in "()"] 
    values = [1 if char == '(' else -1 for char in parentheses]
    
    cumulative_sum = []
    current_sum = 0
    for value in values:
        current_sum += value
        cumulative_sum.append(current_sum)
    
    max_depth = max(cumulative_sum, default=0)
    
    is_valid = all(x >= 0 for x in cumulative_sum)
    
    is_balanced = cumulative_sum[-1] == 0 if cumulative_sum else True
    
    return max_depth, is_valid, is_balanced

# Test lisp-strengen
lisp = '''
(defun LaTeX-newline ()
  "Start a new line potentially staying within comments.
This depends on `LaTeX-insert-into-comments'."
  (interactive)
  (if LaTeX-insert-into-comments
      (cond ((and (save-excursion (skip-chars-backward " \t") (bolp))
                  (save-excursion
                    (skip-chars-forward " \t")
                    (looking-at (concat TeX-comment-start-regexp "+"))))
             (beginning-of-line)
             (insert (buffer-substring-no-properties
                      (line-beginning-position) (match-end 0)))
             (newline))
            ((and (not (bolp))
                  (save-excursion
                    (skip-chars-forward " \t") (not (TeX-escaped-p)))
                  (looking-at
                   (concat "[ \t]*" TeX-comment-start-regexp "+[ \t]*"))))
             (delete-region (match-beginning 0) (match-end 0))
             (indent-new-comment-line))
            ;; `indent-new-comment-line' does nothing when
            ;; `comment-auto-fill-only-comments' is non-nil, so we
            ;; must be sure to be in a comment before calling it.  In
            ;; any other case `newline' is used.
            ((TeX-in-comment)
             (indent-new-comment-line))
            (t
             (newline)))
    (newline)))
'''

# Analyse lisp-strengen
result = check_parentheses(lisp)
print('max_depth:', result[0], 'valid:', result[1], 'balanced:', result[2])