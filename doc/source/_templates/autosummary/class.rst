{{ fullname }}
-------------------------------------------------------------------------------

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   
{% block methods %}
{% if methods %}
   
   .. autosummary::
      :toctree:

   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
{% endif %}
{% endblock %}
