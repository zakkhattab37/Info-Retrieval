
import random

def multiple_box_shadow(n):
    value = []
    for _ in range(n):
        value.append(f"{random.randint(0, 2000)}px {random.randint(0, 2000)}px #FFF")
    return ", ".join(value)

shadows_small = multiple_box_shadow(700)
shadows_medium = multiple_box_shadow(200)
shadows_big = multiple_box_shadow(100)

css_content = f"""
/* Parallax Stars CSS */
#stars {{
  width: 1px;
  height: 1px;
  background: transparent;
  box-shadow: {shadows_small};
  animation: animStar 50s linear infinite;
  position: fixed;
  top: 0;
  left: 0;
  z-index: -3;
}}

#stars:after {{
  content: " ";
  position: absolute;
  top: 2000px;
  width: 1px;
  height: 1px;
  background: transparent;
  box-shadow: {shadows_small};
}}

#stars2 {{
  width: 2px;
  height: 2px;
  background: transparent;
  box-shadow: {shadows_medium};
  animation: animStar 100s linear infinite;
  position: fixed;
  top: 0;
  left: 0;
  z-index: -2;
}}

#stars2:after {{
  content: " ";
  position: absolute;
  top: 2000px;
  width: 2px;
  height: 2px;
  background: transparent;
  box-shadow: {shadows_medium};
}}

#stars3 {{
  width: 3px;
  height: 3px;
  background: transparent;
  box-shadow: {shadows_big};
  animation: animStar 150s linear infinite;
  position: fixed;
  top: 0;
  left: 0;
  z-index: -1;
}}

#stars3:after {{
  content: " ";
  position: absolute;
  top: 2000px;
  width: 3px;
  height: 3px;
  background: transparent;
  box-shadow: {shadows_big};
}}

@keyframes animStar {{
  from {{ transform: translateY(0px); }}
  to {{ transform: translateY(-2000px); }}
}}
"""

with open("stars.css", "w") as f:
    f.write(css_content)
    
print("CSS written to stars.css")
