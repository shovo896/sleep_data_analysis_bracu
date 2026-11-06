%%
%% This is file `cas-dc.cls'.
%%
%% This file is part of the 'CAS Bundle'.
%% ......................................
%% 
%% It may be distributed under the conditions of the LaTeX Project Public
%% License, either version 1.2 of this license or (at your option) any
%% later version. The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% The list of all files belonging to the 'CAS Bundle' is
%% given in the file `manifest.txt'.
%% 

%% $Id: cas-dc.cls 49 2020-03-14 09:05:10Z rishi $
%% $Id: cas-dc.cls    2020-05-22          ozturk $

 \def\RCSfile{cas-dc}%
 %\def\RCSversion{2.1}%
 %\def\RCSdate{2020/03/14}%
 \def\RCSversion{2.2}% Formatting and modifying the class and style files
 \def\RCSdate{2022/05/22}%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{\RCSfile}[\RCSdate, \RCSversion: Formatting class
   for CAS double column articles]
%
\def\ABD{\AtBeginDocument}
%
% switches
% 
\newif\iflongmktitle    \longmktitlefalse
\newif\ifdc             \global\dctrue
\newif\ifsc             \global\scfalse
\newif\ifcasreviewlayout  \global\casreviewlayoutfalse
\newif\ifcasfinallayout   \global\casfinallayoutfalse

\newcounter{blind}
\setcounter{blind}{0}

\def\blstr#1{\gdef\@blstr{#1}}
\def\@blstr{1}
\newdimen\@bls
\@bls=\baselineskip

\DeclareOption{singleblind}{\setcounter{blind}{1}}
\DeclareOption{doubleblind}{\setcounter{blind}{2}}
\DeclareOption{longmktitle}{\global\longmktitletrue}
\DeclareOption{final}{\global\casfinallayouttrue}
\DeclareOption{review}{\global\casreviewlayouttrue}

\ExecuteOptions{a4paper,10pt,twoside,fleqn,review}			
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass{article}

\RequirePackage{graphicx} 
\RequirePackage{amsmath,amsfonts,amssymb} 
\allowdisplaybreaks
\usepackage{mathtools} 
\usepackage{mathpazo} 


\RequirePackage{expl3,xparse}
\@ifundefined{regex_match:nnTF}{\RequirePackage{l3regex}}{}
\RequirePackage{etoolbox,balance}
\RequirePackage{booktabs,makecell,multirow,array,colortbl,dcolumn,stfloats}
\RequirePackage{xspace,xstring}
\RequirePackage[svgnames,dvipsnames]{xcolor}
\usepackage{etoolbox}
\PassOptionsToPackage{pdfpagelabels=true,bookmarks=true,hyperfootnotes=false,hyperindex=true}{hyperref}
\RequirePackage[]{hyperref,cite,url} 
\usepackage{twoopt}
\usepackage[all]{hypcap}
\usepackage[none]{hyphenat}  
\colorlet{scolor}{black}
\colorlet{hscolor}{DarkSlateGrey}
\definecolor{CeruleanRef}{RGB}{0,128,173} 
\hypersetup{
  colorlinks,
  pdftitle={\@title},	
  pdfauthor={\csuse{__short_authors:}}, 
  %pdfsubject={\csuse{__journal_name:}},
  %pdfcreator={LaTeX3; cas-dc.cls; hyperref.sty},
  pdfcreator={Elsevier},
  pdfproducer={pdfTeX},
  pdfkeywords={},
  pdfpagelayout={SinglePage}, 
  pdfview={FitBH},
  pdftoolbar=true,            
  pdfmenubar=true,
  bookmarksopen=true, 
  pdfstartview={FitH},
  breaklinks=true,
  linkcolor={CeruleanRef},
  urlcolor={CeruleanRef},
  citecolor={CeruleanRef},
  linkbordercolor={CeruleanRef},
  filecolor={CeruleanRef},
  menucolor={CeruleanRef},
  allcolors={CeruleanRef},
 }
\usepackage{subcaption}
\usepackage{pdfpages} 
\usepackage[open,
atend,
view={FitH 0},
numbered]{bookmark} 
\RequirePackage[capitalise, nameinlink]{cleveref} 
\Crefname{figure}{Fig.}{Figs.}
\Crefname{equation}{}{} 
\newcommand{\crefpairconjunction}{\color{CeruleanRef}\rmfamily~and~} 
\newcommand*{\crefrangeconjunction}{\color{CeruleanRef}\rmfamily--}    
\renewcommand\UrlFont{\color{CeruleanRef}\rmfamily} 					

\newcommand{\BigAst}{\mathop{\scalebox{1.2}{\raisebox{-0ex}{$\ast$}}}} 		
\newcommand{\SmAst}{\mathop{\scalebox{1.2}{\raisebox{-0.25ex}{$\ast$}}}} 		
\setlength\heavyrulewidth{0.1ex} 					
\setlength\lightrulewidth{0.1ex} 					

\RequirePackage[anythingbreaks]{breakurl} 
\RequirePackage{xurl} 
\usepackage{lipsum} 
\RequirePackage[flushmargin]{footmisc}
\usepackage{fancyhdr}
\pagestyle{fancy}
\RequirePackage{setspace}
\usepackage{soul}
%\usepackage{enumitem} % Original enumaration package
\RequirePackage{tabularx}   
\RequirePackage{ragged2e} 
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}

\renewcommand{\hangfootparindent}{1mm}
\renewcommand{\hangfootparskip}{6pt}
\renewcommand{\footnotemargin}{0.3cm}
\renewcommand{\footnotelayout}{\hspace{1mm}}
\addtolength{\footnotesep}{0.5mm} 		
\renewcommand\footnoterule{\kern0pt \hrule width 1.25cm \kern 0pt\vskip 2pt}	
\setlength{\skip\footins}{0.5cm} 	
\renewcommand{\footnotelayout}{\setstretch{1.1}}	

\let\comma\@empty
\let\tnotesep\@empty
\let\@title\@empty
%
% Load common items
%

%======================================= HYPHENATION SETTINGS =======================================
% Lower values hyphenate the words, larger values hesitate to hyphenate the word
% Lower the hyphenpenalty value more hyphenation occurs
\hyphenpenalty=10
\exhyphenpenalty=10
\widowpenalty=100
\clubpenalty=500
\flushbottom 			

\RequirePackage{cas-common}

%
% Specific to Single Column
%
\ExplSyntaxOn

\RenewDocumentCommand \maketitle { }
{
  \ifbool { usecasgrabsbox }
    { 
      \setcounter{page}{0}
      \thispagestyle{empty}
      \unvbox\casgrabsbox  
    } { }  
  \pagebreak  
  \ifbool { usecashlsbox }
    { 
      \setcounter{page}{0}
      \thispagestyle{empty}
      \unvbox\casauhlbox
    } { }    
  \pagebreak
  \thispagestyle{first}    
  \ifbool{longmktitle}
  {
    \LongMaketitleBox
    \ProcessLongTitleBox
  }
  {
    \twocolumn[\MaketitleBox]  
    \printFirstPageNotes
  }
  \setcounter{footnote}{\int_use:N \g_stm_fnote_int}
  \renewcommand\thefootnote{\arabic{footnote}}
  \gdef\@pdfauthor{\infoauthors}
  \gdef\@pdfsubject{Engineering\ Science\ and\ Technology,\ an\ International\ Journal\ xx\ (20xx)\ xxx-xxx} 
}

%
% Fonts
%
\RequirePackage[T1]{fontenc}

%\file_if_exist:nTF { stix.sty }
%{
  \file_if_exist:nTF { charis.sty }
  {

  }
  { 
    \RequirePackage{charter} 											
    \RequirePackage{helvet}
    \setlength\parindent{1.25em} 										
    \setlength{\parskip}{0.5pt} 										
  }
%}
{
  \iow_term:x {  *********************************************************** }
  \iow_term:x { ~Stix ~ and ~ Charis~ fonts ~ are ~ not ~ available ~ }
  \iow_term:x { ~ in ~TeX~system.~Hence~CMR~ fonts~ are ~ used. }
  \iow_term:x {  *********************************************************** }
}

\file_if_exist:nTF { inconsolata.sty } 		
{ \RequirePackage[scaled=.85]{inconsolata} }	
{ \tex_gdef:D \ttdefault { cmtt } } 

\ExplSyntaxOff

%
% Page geometry
%

\usepackage[ 
   twoside,  			
   showframe=false, 	
   paperwidth=210mm,
   paperheight=280mm,
   vmargin={23mm,27mm},
   inner=11.5mm,
   outer=15mm,
   headsep=12pt,	
   footskip=12pt, 
   columnsep=18pt 	
]{geometry}

\makeatletter
\renewcommand{\@biblabel}[1]{\hskip -1pt[#1]}
\makeatother

\endinput

%
% End of class 'cas-dc'
%


%
% End of class 'cas-dc'
%


%


%
% End of class 'cas-dc'
%

nd of class 'cas-dc'
%

