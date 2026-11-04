"""Bounded-memory canonical manifest reader for independently verifying cloud outcomes."""
import json

def manifest_items(source,chunk_size=65536,max_item_chars=4000000):
 if chunk_size<1 or max_item_chars<1:raise ValueError('Positive reader limits required')
 buffer='';position=0;eof=False;decoder=json.JSONDecoder()
 def fill():
  nonlocal buffer,position,eof
  buffer=buffer[position:];position=0
  part=source.read(chunk_size)
  if part:buffer+=part
  else:eof=True
 def char():
  nonlocal position
  while True:
   while position<len(buffer) and buffer[position].isspace():position+=1
   if position<len(buffer):return buffer[position]
   if eof:raise ValueError('Truncated manifest')
   fill()
 def consume(expected):
  nonlocal position
  if char()!=expected:raise ValueError('Malformed manifest delimiter')
  position+=1
 def value():
  nonlocal position
  char();start=position
  while True:
   try:
    result,end=decoder.raw_decode(buffer,position)
    # A number at a buffer boundary may continue in the next chunk.
    if not eof and (end==len(buffer) or buffer[end] not in ',]}: \t\r\n'):fill();start=position;continue
    if end-start>max_item_chars:raise ValueError('Manifest item exceeded bounded limit')
    position=end;return result
   except json.JSONDecodeError as exc:
    if eof:raise ValueError('Truncated or malformed manifest value') from exc
    if len(buffer)-position>max_item_chars:raise ValueError('Manifest item exceeded bounded limit') from exc
    fill();start=position
 consume('{')
 empty=char()=='}'
 if empty:consume('}')
 while not empty:
  key=value()
  if not isinstance(key,str):raise ValueError('Manifest string key required')
  consume(':')
  if key in ('candidates','rejections'):
   consume('[')
   if char()!=']':
    while True:
     row=value()
     if not isinstance(row,dict):raise ValueError('Manifest outcome object required')
     yield key,row
     if char()==']':break
     consume(',')
   consume(']')
  else:yield key,value()
  if char()=='}':consume('}');break
  consume(',')
 # No unparsed trailing data is permitted.
 while position<len(buffer):
  if not buffer[position].isspace():raise ValueError('Trailing manifest data')
  position+=1
 for part in iter(lambda:source.read(chunk_size),''):
  if part.strip():raise ValueError('Trailing manifest data')
