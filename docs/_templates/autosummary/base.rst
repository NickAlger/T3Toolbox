{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{% if objtype == 'function' %}
.. autofunction:: {{ objname }}
{% elif objtype == 'class' %}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
{% else %}
.. auto{{ objtype }}:: {{ objname }}
{% endif %}

